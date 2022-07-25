
import os
import json
import math
import argparse
import torch
import random
import numpy as np

from collections import defaultdict

from tqdm import tqdm
from datetime import datetime as dt
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from collections import Counter, OrderedDict

from models.bunt_model import Encoder, Decoder, Bunt
from utils import random_seed, save_dependencies, last_commit_msg

from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

MORE = 4
MIN = 1e-6
MAX = 1e6
LARGE = 4
TER = 20
SIZE = 8

torch.multiprocessing.set_sharing_strategy('file_system')

# --------
# Dataset 
# --------


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, args):
        self.args = args
        self.mode = mode

        self.train = json.load(
            open(os.path.join(path, "processed", "user_bundle_offline.json"), "r"))
        self.train = {int(u): self.train[u] for u in self.train}

        self.item_set = json.load(
            open(os.path.join(path, "raw", "item_id_lookup.json"), "r"))
        self.item_size = self.pad_token = len(self.item_set)

        self.eos_token = len(self.item_set) + 1
        self.bos_token = len(self.item_set) + 2
        self.mask_token = len(self.item_set) + 3

        if self.mode == 'train':
            # mask one of the bundle for training
            self.users = [u for u in self.train if len(self.train[u]) >= 2]
        else:
            assert self.mode == 'valid' or self.mode == 'test', 'mode should be `valid` or `test`'
            self.data = json.load(
                open(os.path.join(path, "processed", f"user_bundle_{self.mode}.json"), "r"))
            self.data = {int(u): self.data[u][0] for u in self.data}
            self.users = list(self.data.keys())

        self.item2cateid, self.cate2id, self.cate_set, self.cate_num, self.max_cate_len = self._load_side(path, tag='cate')
        self.item2attrid, self.attr2id, self.attr_set, self.attr_num, self.max_attr_len = self._load_side(path, tag='attr')
        self.cate_pad_token = self.cate_num
        self.attr_pad_token = self.attr_num
        self.args.cate_reweights = self._reweight(tag='cate')
        self.args.attr_reweights = self._reweight(tag='attr')

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        # basic info
        u = self.users[index]
        train = self.train[u]
        # pure masking
        u, src, trg, par = self._gen_task(u, train)
        # side masking
        cate_pos, cate_neg, cate_trg = self._tag_task(trg, 'cate')
        attr_pos, attr_neg, attr_trg = self._tag_task(trg, 'attr') 
        # return 
        return {
            'u': torch.LongTensor([u]), 'src': torch.LongTensor(src), 'trg': torch.LongTensor(trg), 'par': torch.LongTensor(par),
            'cate_pos': torch.LongTensor(cate_pos), 'cate_neg': torch.LongTensor(cate_neg), 'cate_trg': torch.LongTensor(cate_trg),
            'attr_pos': torch.LongTensor(attr_pos), 'attr_neg': torch.LongTensor(attr_neg), 'attr_trg': torch.LongTensor(attr_trg)
        }

    def _load_side(self, path, tag='cate'):
        # load tag mapping
        item2tagid = json.load(open(os.path.join(path, "raw", f"item_{tag}.json"), "r"))
        item2tagid = {int(i): item2tagid[i] for i in item2tagid}
        # load tag info 
        tag2id = json.load(open(os.path.join(path, "raw", f"{tag}_id_lookup.json"), "r"))
        tag_set = set(tag2id.values())
        tag_num = max(tag_set) + 1
        max_tag_len = max(len(i) for i in item2tagid.values())
        # return
        return item2tagid, tag2id, tag_set, tag_num, max_tag_len

    def _reweight(self, tag='cate'):
        item2tagid = self.item2cateid if tag == 'cate' else self.item2attrid
        tag_pad_token = self.cate_pad_token if tag == 'cate' else self.attr_pad_token

        tag_pool = []
        for u in self.train:
            for b in self.train[u]:
                for i in b:
                    if i in item2tagid:
                        tag_pool.extend(item2tagid[i])

        unique_tags = list(np.unique(tag_pool))
        temp_tag_reweights = list(compute_class_weight(class_weight='balanced', classes=unique_tags, y=tag_pool))

        reweights = [0] * (tag_pad_token + 1)
        for i, w in zip(unique_tags, temp_tag_reweights):
            reweights[i] = w
        return torch.Tensor(reweights).cuda()

    def _gen_task(self, u, train):
        inputs = []
        if self.mode == 'train':
            # leave one out
            i = random.randint(0, len(train)-1)
            for j, items in enumerate(train):
                if i != j:
                    inputs.append([self.mask_token] + items[:SIZE] +
                                  [self.pad_token]*(SIZE-len(items)))
            # target or mask_target
            target = train[i][:self.args.max_size] + [self.eos_token]
            cut_length = random.randint(1, len(target))
            target = [self.bos_token] + target[:cut_length]
            par = target.copy()
            for _ in range(self.args.delta_len):
                mask_id = random.randint(1, cut_length)
                par[mask_id] = self.mask_token

        else:
            # user all user history
            for items in train:
                inputs.append([self.mask_token] + items[:SIZE] +
                              [self.pad_token]*(SIZE-len(items)))
            # target from valid or test
            target =  self.data[u][:self.args.max_size]
            target = [self.bos_token] + target + \
                [self.pad_token] * (self.args.max_size - len(target))
            par = [self.pad_token]

        return u, inputs, target, par

    def _tag_task(self, target, tag='cate'):
        # tag info
        item2tagid = self.item2cateid if tag == 'cate' else self.item2attrid
        tag_pad_token = self.cate_pad_token if tag == 'cate' else self.attr_pad_token
        tag_set = self.cate_set if tag == 'cate' else self.attr_set
        max_tag_len = self.max_cate_len if tag == 'cate' else self.max_attr_len
        # tag flag
        pos_flag = args.is_cate_pos if tag == 'cate' else args.is_attr_pos
        neg_flag = args.is_cate_neg if tag == 'cate' else args.is_attr_neg
        # pos samples
        mask_set = set([i for i in range(len(target)) if random.uniform(0, 1) <= self.args.pos_mask_prob]) if pos_flag else set([])
        tag_pos = [random.choice(item2tagid[i]) if i in item2tagid and idx not in mask_set and len(
            item2tagid[i]) else tag_pad_token for idx, i in enumerate(target)]
        # neg samples
        mask_set = set([i for i in range(len(target)) if random.uniform(0, 1) <= self.args.neg_mask_prob]) if neg_flag else set([])
        tag_neg = [random.choice(list(tag_set - set(item2tagid[i])))
                   if i in item2tagid and idx not in mask_set else tag_pad_token for idx, i in enumerate(target)]
        # targets
        tag_target = [item2tagid[t] + [tag_pad_token] * (max_tag_len - len(
                item2tagid[t])) if t in item2tagid else [tag_pad_token] * max_tag_len for t in target]
        return tag_pos, tag_neg, tag_target

# ---------------------
# Evaluation
# ---------------------


def evaluate(model, val_iter, dataset=None, args=None, mode='test'):
    with torch.no_grad():
        model.eval()
        metrics = defaultdict(list)
        instances = {}
        preds = []

        for b, batch in tqdm(enumerate(val_iter)):
            u, src, trg = batch['u'], batch['src'].cuda(), batch['trg'].cuda()
            cate_pos, attr_pos, cate_trg, attr_trg = \
                batch['cate_pos'].cuda(), batch['attr_pos'].cuda(), batch['cate_trg'].cuda(), batch['attr_trg'].cuda()

            # pure seq
            output, _, _, _ = model.generate(src, delta_len=args.max_size)

            for i in range(trg.size(1)):
                trgs = set(trg[1:, i].tolist()) - {args.pad, args.eos, args.bos}

                pred = []
                for j in output[1:, i].max(1)[-1].tolist():
                    if j in {args.pad, args.bos}: continue
                    elif j == args.eos: break 
                    else: pred.append(j)

                instances[u[i].item()] = {
                    'all_set': list(pred),
                    'target_set': list(trgs),
                    'partial_set': list(trgs & set(pred))
                }

                pred = set(pred)

                metrics['acc'].append(len(trgs & pred) / len(trgs | pred))
                metrics['prec'].append(len(trgs & pred) / len(pred) if len(pred) else 0)
                metrics['rec'].append(len(trgs & pred) / len(trgs))

                preds.append(tuple(sorted(list(pred))))

            # side seq
            side_output, cate_output, attr_output, conv_output = model.generate(src, delta_len=args.max_size, cate_pos=cate_pos, attr_pos=attr_pos)

            ## predict cate & attr
            _, cate_acc = side_loss_and_acc(cate_output.view(-1, cate_output.size(-1)), cate_trg.view(-1, cate_trg.size(-1)), pad=args.cate_pad, reweights=args.cate_reweights)
            metrics['cate_acc'].append(cate_acc)
            _, attr_acc = side_loss_and_acc(attr_output.view(-1, attr_output.size(-1)), attr_trg.view(-1, attr_trg.size(-1)), pad=args.attr_pad, reweights=args.attr_reweights)
            metrics['attr_acc'].append(attr_acc)

            ## predict conv
            cmp_conv_output = conv_output.view(-1, conv_output.size(-1)).squeeze(dim=-1)
            item_hits, side_hits = create_conv_label(side_output.view(-1, side_output.size(-1)), trg.flatten(), cate_output.view(-1, cate_output.size(-1)), \
                    cate_trg.view(-1, cate_trg.size(-1)), cate_pos.flatten(), attr_output.view(-1, attr_output.size(-1)), attr_trg.view(-1, attr_trg.size(-1)), attr_pos.flatten())

            valid_cmp_conv = cmp_conv_output[(item_hits + side_hits) != 0].argmax(dim=-1).tolist() 
            valid_label = item_hits[(item_hits + side_hits) != 0].tolist()
            metrics['conv_ratio'].append(np.mean(valid_cmp_conv) if len(valid_cmp_conv) else 0)
            metrics['conv_acc'].append(accuracy_score(valid_label, valid_cmp_conv))
            metrics['conv_f1'].append(f1_score(valid_label, valid_cmp_conv)) 

            ## predict items
            for i in range(trg.size(1)):
                trgs = set(trg[1:, i].tolist()) - {args.pad, args.eos, args.bos}

                pred = set([])
                for j in side_output[1:, i].max(1)[-1].tolist():
                    if j in {args.pad, args.bos}: continue
                    elif j == args.eos: break 
                    else: pred.add(j)

                metrics['side_acc'].append(len(trgs & pred) / len(trgs | pred))
                metrics['side_prec'].append(len(trgs & pred) / len(pred) if len(pred) else 0)
                metrics['side_rec'].append(len(trgs & pred) / len(trgs))

        metrics = OrderedDict({i: np.mean(metrics[i]) for i in metrics})
        metrics['f1'] = (2 * metrics['rec'] * metrics['prec'] / (metrics['rec'] + metrics['prec']))
        return metrics, instances

# ---------------------
# Training
# ---------------------


def train(e, model, optimizer, train_iter, args):
    model.train()
    metrics = defaultdict(list)
    tqdm_iter = tqdm(train_iter)
    for b, batch in enumerate(tqdm_iter):
        src, trg, par = batch['src'].cuda(), batch['trg'].cuda(), batch['par'].cuda()
        cate_pos, cate_trg, attr_pos, attr_trg = \
            batch['cate_pos'].cuda(), batch['cate_trg'].cuda(), batch['attr_pos'].cuda(), batch['attr_trg'].cuda()

        optimizer.zero_grad()

        # generation loss
        output, cate_output, attr_output, conv_output = model(src, par)

        # normal
        cmp_output = output[par == args.mask]
        cmp_target = trg[par == args.mask]

        loss = F.nll_loss(cmp_output, cmp_target, ignore_index=args.pad)
        metrics['acc'].append((cmp_output.max(dim=-1)[-1] == cmp_target).float().mean().item())

        # tags
        side_output, cate_output, attr_output, conv_output = model(src, par, cate_pos=cate_pos, attr_pos=attr_pos)

        cate_loss, cate_acc = side_loss_and_acc(cate_output[par==args.mask], cate_trg[par==args.mask], pad=args.cate_pad, reweights=args.cate_reweights)
        attr_loss, attr_acc = side_loss_and_acc(attr_output[par==args.mask], attr_trg[par==args.mask], pad=args.attr_pad, reweights=args.attr_reweights)
        
        metrics['cate_acc'].append(cate_acc)
        metrics['attr_acc'].append(attr_acc)

        side_loss = F.nll_loss(side_output[par == args.mask], trg[par == args.mask], ignore_index=args.pad)

        cmp_conv_output = conv_output[par == args.mask].squeeze(dim=-1)
        item_hits, side_hits = create_conv_label(side_output[par == args.mask], trg[par == args.mask], cate_output[par==args.mask], \
                cate_trg[par==args.mask], cate_pos[par==args.mask], attr_output[par==args.mask], attr_trg[par==args.mask], attr_pos[par==args.mask])
        conv_loss = (- (item_hits * cmp_conv_output[:, 1]) - (1-item_hits) * cmp_conv_output[:, 0])[(item_hits + side_hits) != 0].sum() / (((item_hits + side_hits) != 0).sum() + MIN)

        valid_cmp_conv = cmp_conv_output[(item_hits + side_hits) != 0].argmax(dim=-1).tolist() 
        valid_label = item_hits[(item_hits + side_hits) != 0].tolist()
        metrics['conv_ratio'].append(np.mean(valid_cmp_conv) if len(valid_cmp_conv) else 0)
        metrics['conv_acc'].append(accuracy_score(valid_label, valid_cmp_conv))
        metrics['conv_f1'].append(f1_score(valid_label, valid_cmp_conv))

        # multi loss
        loss += side_loss + args.cate_coeff * cate_loss + args.attr_coeff * attr_loss + args.conv_coeff * conv_loss
        loss.backward()

        optimizer.step()

        metrics['loss'].append(loss.item())

        tqdm_iter.set_description(f"[Epoch {e}] " + " | ".join([m + ':' + f'{np.mean(metrics[m]):.4f}' for m in metrics]))

def side_loss_and_acc(output, trg, pad, reweights):
    # loss
    trg_ = torch.zeros(trg.size(0), pad+1, device=trg.device)
    trg_[torch.arange(trg.size(0)).unsqueeze(-1), trg] = 1
    trg_ = trg_ * reweights.unsqueeze(dim=0)
    loss = - (output * trg_).sum(dim=-1).mean()
    # acc
    acc, num = 0, 0
    for p, t in zip(output.argmax(dim=-1).tolist(), trg):
        t = set(t.tolist()) - {pad}
        acc += int(p in t)
        num += int(len(t) > 0)
    return loss, acc / num

def create_conv_label(item_output, item_target, cate_output, cate_target, cate_exists, attr_output, attr_target, attr_exists):
    # item 
    item_pred = item_output.argmax(dim=-1) # (b, )
    item_hits = (item_target == item_pred) # (b, )
    # cate
    cate_output[torch.arange(cate_exists.size(0)), cate_exists] = - MAX
    cate_pred = cate_output.argmax(dim=-1) # (b, )
    cate_hits = (cate_pred != args.cate_pad) & (cate_pred.unsqueeze(dim=-1) == cate_target).sum(dim=-1).bool()
    # attr
    attr_output[torch.arange(attr_exists.size(0)), attr_exists] = - MAX
    attr_pred = attr_output.argmax(dim=-1) # (b, )
    attr_hits = (attr_pred != args.attr_pad) & (attr_pred.unsqueeze(dim=-1) == attr_target).sum(dim=-1).bool()    
    return item_hits.float(), (cate_hits | attr_hits).float()

# ---------------------
# Main Function
# ---------------------


def main(data_path, args, pretrained_weights=None):
    terminate_cnt = 0

    def custom_collate(x):
        src = pad_sequence([item['src'][:args.max_size] for item in x], padding_value=args.pad)
        src[..., -1] = args.mask
        return {
            # pure sequence
            'u': [item['u'] for item in x],
            'src': src,
            'trg': pad_sequence([item['trg'] for item in x], padding_value=args.pad),
            'par': pad_sequence([item['par'] for item in x], padding_value=args.pad),
            # cate sequence
            'cate_pos': pad_sequence([item['cate_pos'] for item in x], padding_value=args.cate_pad),
            'cate_neg': pad_sequence([item['cate_neg'] for item in x], padding_value=args.cate_pad),
            'cate_trg': pad_sequence([item['cate_trg'] for item in x], padding_value=args.cate_pad),
            # attr sequence
            'attr_pos': pad_sequence([item['attr_pos'] for item in x], padding_value=args.attr_pad),
            'attr_neg': pad_sequence([item['attr_neg'] for item in x], padding_value=args.attr_pad),
            'attr_trg': pad_sequence([item['attr_trg'] for item in x], padding_value=args.attr_pad),
        }

    print("[!] loading dataset...")
    train_dataset = Dataset(path=data_path, mode="train", args=args)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=True, num_workers=32)
    valid_data = DataLoader(Dataset(path=data_path, mode="valid", args=args),
                            batch_size=args.val_batch_size, collate_fn=custom_collate, shuffle=True, num_workers=32)
    test_data = DataLoader(Dataset(path=data_path, mode="test", args=args),
                           batch_size=args.val_batch_size, collate_fn=custom_collate, shuffle=True, num_workers=32)

    args.item_size = train_dataset.item_size
    args.item_large = args.item_size + MORE
    # pure sequence
    args.pad = train_dataset.pad_token
    args.mask = train_dataset.mask_token
    args.eos = train_dataset.eos_token
    args.bos = train_dataset.bos_token
    # side
    args.cate_pad = train_dataset.cate_pad_token
    args.attr_pad = train_dataset.attr_pad_token

    print("[!] Instantiating models...")

    encoder = Encoder(args)
    decoder = Decoder(args)
    model = Bunt(encoder, decoder, args).cuda()

    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    print(model)

    val_metrics, _ = evaluate(model, valid_data, train_dataset, args)
    best_f1 = val_metrics['f1']
    print("[Epoch 0] " + " | ".join([m + ':' +
          f'{val_metrics[m]:.4f}' for m in val_metrics]))

    test_metrics, instances = evaluate(model, test_data, train_dataset, args)
    print("[Epoch 0] " + " | ".join([m + ':' +
          f'{test_metrics[m]:.4f}' for m in test_metrics]))

    for e in range(1, args.epochs+1):
        train(e, model, optimizer, train_data, args)

        if e % args.print_every == 0:

            # Val loss
            val_metrics, instances = evaluate(
                model, valid_data, train_dataset, args)
            val_f1 = val_metrics['f1']
            print(f"[Epoch {e}] " + " | ".join([m + ':' +
                  f'{val_metrics[m]:.4f}' for m in val_metrics]))

            print("[!] saving model...")
            torch.save(model.state_dict(), os.path.join(
                ckpt_dir, f'model_{e}.pt'))

            # Test loss
            test_metrics, _ = evaluate(model, test_data, train_dataset, args)
            print(f"[Epoch {e}] " + " | ".join([m + ':' +
                  f'{test_metrics[m]:.4f}' for m in test_metrics]))

            # Save the model if the validation loss is the best we've seen so far.

            if not best_f1 or val_f1 > best_f1:
                best_f1 = val_f1
                with open(os.path.join(ckpt_dir, "test.log"), "w") as f:
                    test_metrics.update({'epoch': e})
                    f.write(json.dumps(test_metrics, indent=2))
                terminate_cnt = 0
            else:
                terminate_cnt += 1

        # early stop
        if terminate_cnt == TER:
            break

# ----------------
# Arguments
# ----------------


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    # data
    p.add_argument('--data', type=str, default='steam',
                   help='steam')
    # model
    p.add_argument('--n_layers', type=int, default=1,
                   help='layer number of model')
    p.add_argument('--n_heads', type=int, default=4,
                   help='head number of model')
    p.add_argument('--embed_size', type=int, default=32,
                   help='embed size for items')
    # training (normal)
    p.add_argument('--seed', type=int, default=42,
                   help='random seed for model training')
    p.add_argument('--epochs', type=int, default=500,
                   help='number of epochs for train')
    p.add_argument('--batch_size', type=int, default=64,
                   help='number of epochs for train')
    p.add_argument('--val_batch_size', type=int, default=128,
                   help='number of epochs for validation / testing')
    p.add_argument('--lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('--delta_len', type=int, default=2,
                   help='slot size K')
    p.add_argument('--dropout', type=float, default=0,
                   help='model dropout')
    p.add_argument('--max_size', type=int, default=20,
                   help='maximum bundle size')
    p.add_argument('--decay', type=float, default=0,
                   help='value of weight decay')
    # side info
    p.add_argument('--is_cate_pos', type=int, default=1,
                   help='whether add cate_pos to training')
    p.add_argument('--is_cate_neg', type=int, default=0,
                   help='whether add attr_neg to training')
    p.add_argument('--is_attr_pos', type=int, default=1,
                   help='whether add attr_pos to training')
    p.add_argument('--is_attr_neg', type=int, default=0,
                   help='whether add attr_neg to training')
    p.add_argument('--pos_mask_prob', type=float, default=0.5,
                   help='pos tags masking probability')
    p.add_argument('--cate_coeff', type=float, default=0.1,
                   help='coefficient of masked category generation loss')
    p.add_argument('--attr_coeff', type=float, default=0.1,
                   help='coefficient of masked attribute generation loss')
    p.add_argument('--conv_coeff', type=float, default=0.1,
                   help='coefficient of masked conv output loss')
    # logging
    p.add_argument('--device', type=str, default='cuda:0',
                   help='cuda:x or cpu')
    p.add_argument('--ckpt_dir', type=str, default='test',
                   help='checkpoint saving directory')
    p.add_argument('--load_pretrained_weights', type=str, default=None,
                   help='checkpoint directory to load')
    p.add_argument('--print_every', type=float, default=10,
                   help="print evaluate results every X epoch")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.seed = random_seed(args.seed)

    # logging folder
    branch, commit = last_commit_msg()
    ckpt_dir = os.path.join('checkpoints', branch, commit, args.ckpt_dir, f'seed_{args.seed}_{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}')

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with open(os.path.join(ckpt_dir, "args.log"), "w") as f:
        f.write(json.dumps(vars(args), indent=2))
    save_dependencies(ckpt_dir)

    # main
    print(f"set ckpt as {ckpt_dir}")
    main(f"data/{args.data}", args, args.load_pretrained_weights)
    print(f"set ckpt as {ckpt_dir}")
