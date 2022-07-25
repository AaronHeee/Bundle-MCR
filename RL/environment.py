import os 
import json 
import random 
import numpy as np

from tkinter import _flatten

from torch._C import AggregationType

import gym
from gym import spaces

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from models.bunt_model import Encoder, Decoder, Bunt
from collections import OrderedDict, Counter, defaultdict

from stable_baselines3.common.utils import obs_as_tensor
from .simulator import SimulatedUser
from .helper import Helper
from .utils import f1_score, recall_score, precision_score, accuracy_score, metrics
from tqdm import tqdm

MORE = 4
MAX_LEN = 64
SIZE = 8



# Dataset

class NextBundleDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, args):
        self.args = args
        self.mode = mode
        
        self.offline = json.load(open(os.path.join(path, "processed", "user_bundle_offline.json"), "r"))
        self.offline = {int(u):self.offline[u] for u in self.offline}

        self.item_set = json.load(open(os.path.join(path, "raw", "item_id_lookup.json"), "r"))
        self.item_size = self.pad_token = len(self.item_set)
        self.eos_token = len(self.item_set) + 1
        self.bos_token = len(self.item_set) + 2
        self.mask_token = len(self.item_set) + 3
    
        assert self.mode == 'valid' or self.mode == 'test' or self.mode == 'online', 'mode should be `valid` or `test` or `online`'
        self.data = json.load(open(os.path.join(path, "processed", f"user_bundle_{self.mode}.json"), "r"))
        self.data = {int(u):self.data[u][0] for u in self.data} # [[xxx]] -> [xxx]
        self.users = list(self.data.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.sample(index)

    def sample(self, index):
        u = self.users[index]
        offline = self.offline[u]

        inputs = []
        for items in offline:
            inputs.append([self.mask_token] + items[:SIZE] + [self.pad_token]*(SIZE-len(items)))

        target = [self.bos_token] + self.data[u] + [self.eos_token] 
        return torch.LongTensor([u]), torch.LongTensor(inputs), torch.LongTensor(target)

# Environment

class BUNDLERecommendEnv():

    def __init__(self, args, mode="online"):
        super(BUNDLERecommendEnv, self).__init__()

        self.args = args
        self.data = args.data
        self.infos = {}
        self.command = args.command 
        self.seed = args.seed
        self.max_run = args.max_run 
        self.eval_max_run = args.eval_max_run if mode != 'online' else args.max_run
        self.device = args.device

        # set state dict
        self.history_dict = {
            'ask_single_suc': 1,
            'ask_single_fail': 2,
            'rec_single_suc': 3,
            'rec_single_fail': 4,
            'rec_bundle_suc': 5,
            'until_max': 6
        }

        # load data
        print("[!] Loading dataset...")

        data_path = f"data/{args.data}"
        self.dataset = NextBundleDataset(path=data_path, mode=mode, args=args)

        # helper for tag
        self.cate_helper = Helper(path=data_path, mode='cate')
        self.cate_pad = args.cate_pad = self.cate_helper.pad
        self.attr_helper = Helper(path=data_path, mode='attr')
        self.attr_pad = args.attr_pad = self.attr_helper.pad

        # load item info
        item_size = args.item_size = self.dataset.item_size
        args.item_large = item_size + MORE
        pad = args.pad = self.dataset.pad_token
        eos = args.eos = self.dataset.eos_token
        bos = args.bos = self.dataset.bos_token
        mask = args.mask = self.dataset.mask_token
        self.pad, self.eos, self.bos, self.mask = pad, eos, bos, mask

        # load model
        print("[!] Instantiating models...")

        encoder = Encoder(args)
        decoder = Decoder(args)
        self.bundle_rec = Bunt(encoder, decoder, args).cuda()

        if args.pretrained_weights is not None:
            self.bundle_rec.load_state_dict(torch.load(args.pretrained_weights))

        if args.load_ckpt_dir is not None:
            self.bundle_rec.load_state_dict(torch.load(os.path.join(args.load_ckpt_dir, "bunt.pt")))

        # simiulated user
        self.kp_model = SimulatedUser(path=data_path)

        # initialization
        self.action_space = spaces.Discrete(2)

        # load conv policies with pretrained network
        self.conv_policy = args.conv_policy_class(args=self.args)
        self.conv_policy.policy_net_2 = self.bundle_rec.conv_predictor
        self.conv_policy.policy_net_2.requires_grad = True 
        self.conv_policy = self.conv_policy.to(self.args.device)
        self.conv_policy.device = self.args.device
        self.conv_policy.initialize_optimizer()

        if self.args.load_ckpt_dir is not None:
            self.conv_policy.load_state_dict(torch.load(os.path.join(self.args.load_ckpt_dir, "conv_manager.pt")))

        # load side policies with pretrained network
        self.side_policies = {}
        for n, network in [('item_policy', self.bundle_rec.item_predictor), ('cate_policy', self.bundle_rec.cate_predictor), ('attr_policy', self.bundle_rec.attr_predictor)]:
            self.side_policies[n] = args.side_policy_class(embed_size=args.embed_size)
            self.side_policies[n].to(self.args.device)
            self.side_policies[n].policy_net = network
            self.side_policies[n].policy_net.requires_grad = True
            self.side_policies[n].name = n
            self.side_policies[n].initialize_optimizer()

        self.reset()

    def reset(self, index=None):
        # sample or assign an index
        index = random.randint(0, len(self.dataset)-1) if index is None else index
        
        # reset data
        u, src, trg = self.dataset.sample(index)
        self.user, self.src, self.trg = u.item(), src.unsqueeze(1).to(self.device), trg.unsqueeze(1).to(self.device)
        
        # reset list
        self.partial_list, self.target_set, self.all_set = [], set(trg.flatten().tolist()) - {self.bos, self.eos, self.pad}, set([])
        self.black_lists = [[self.bos, self.pad, self.eos] for _ in range(self.args.max_run * self.args.delta_len + 1)]

        # reset user simulator
        self.kp_model.reset()

        # reset cate helper 
        self.cate_helper.reset() 
        self.attr_helper.reset()

        # reset wrap info
        self.wrap_info = defaultdict(list)

        # reset slots
        self.blank_slots = list(range(1, self.args.max_run * self.args.delta_len + 1))
        self.slots = [self.blank_slots.pop(0) for _ in range(self.args.delta_len)]
        self.cur = torch.LongTensor([[self.bos]] + [[self.pad] for _ in range(self.args.max_run * self.args.delta_len)]).to(self.device) # (total_slots + 1, 1)
        
        # reset state
        self.cur_conver_step, self.history_vec = 0, [0] * self.eval_max_run
        return self._get_state()

    def _get_state(self, embed=None, backbone=True):
        if embed is None:
            embed = torch.Tensor([0.] * self.args.embed_size).unsqueeze(0)
        state = {
            "result": torch.LongTensor(list(_flatten(self.history_vec))).unsqueeze(0),
            "embed": embed.cpu().detach()
        }
        if backbone is True:
            self.output_dict = self._get_backbone_output()

        # avg slot vector
        slot = 0
        for i in self.slots:
            slot += self.output_dict['embed'][:, i].cpu().detach()
        slot = slot / (len(self.slots) + 1e-8)
        state.update({'slot': slot})  

        return state # (b, *, *)

    def _get_backbone_output(self):

        cate_pos, _ = self.cate_helper.summarize(k=self.cur.size(0))
        attr_pos, _ = self.attr_helper.summarize(k=self.cur.size(0))

        with torch.no_grad():
            output, cate_output , attr_output, _, output_embedding = self.bundle_rec.propose(self.src, self.cur.clone(), slots=self.slots, cate_pos=cate_pos, attr_pos=attr_pos, black_lists=self.black_lists.copy()) # (total_slots + 1, b, *)
        output_dict = {'item_policy': output.transpose(0, 1), 'cate_policy': cate_output.transpose(0, 1), 'attr_policy': attr_output.transpose(0, 1), 'embed': output_embedding.transpose(0, 1), 'slots': self.slots} # (b, total_slots + 1, *)
        return output_dict

    def step(self, conv_action, conv_state=None, conv_value=None, conv_log_prob=None):
        """
        :return next state: state
        :return conv_record: {
            obs: {k: cpu tensor (1, *)}, 
            reward: float,
            action: cpu tensor (1,),
            value: cpu tesnor (1,),
            log_prob: cpu tensor (1,),
            }
        :return side_records: {
            item_policy / cate_policy / attr_policy: list of record,
        }
        """

        # initialization
        done, reward, ask_sus = 0, 0, 0
        self.delta_record = {'item_policy': [], 'cate_policy': [], 'attr_policy': []}
        self.side_records = {'item_policy': [], 'cate_policy': [], 'attr_policy': []} 

        if conv_action == 0:   # ask feature 

            for s in self.output_dict['slots']:
                embed = self.output_dict['embed'][:, s]
                for mode, helper in [('cate_policy', self.cate_helper), ('attr_policy', self.attr_helper)]:
                    if self.args.policy_strategy == 'random':
                        tags = helper.select(self.args.ask_k, slot=s)
                        actions = values = log_probs = ask_reward = 0
                        obs = None

                    elif self.args.policy_strategy == 'pretrain':
                        # get obs and values
                        obs = self._get_state(embed=embed, backbone=False)
                        with torch.no_grad():
                            _, values, _ = self.side_policies[mode](obs_as_tensor(obs, 'cuda'))
                        # get actions and log_probs 
                        model_output = self.output_dict[mode][:, s]
                        model_output[:, list(helper.pos_tag_per_slot[s]) + list(helper.neg_tag_per_slot[s]) + [helper.pad]] = float('-inf')
                        actions = model_output.argmax(dim=-1)
                        tag = actions.item()
                        log_probs = model_output[:, actions.item()]

                    # update helper and calculate reward
                    t_ans = self.kp_model.user_feedback_slot(tag, self.target_set - set(self.black_lists[s]), mode=mode, slot=s)
                    helper.record(tag=tag, mode=t_ans, slot=s)
                    self.delta_record[mode].append((tag, t_ans))
                    
                    if t_ans == 'neg':
                        for i in range(len(self.black_lists)):
                            self.black_lists[i] = list(set(self.black_lists[i]) | helper.tagid2item[tag])

                    # collect cate and attr_policy info
                    r = int(t_ans == 'pos')
                    self.side_records[mode].append({'obs': obs, 'reward': r, 'action': actions.cpu(), 'value': values.flatten().cpu(), 'log_prob': log_probs.cpu()})
            
            if ask_sus > 0:
                self.history_vec[self.cur_conver_step] = self.history_dict['ask_single_suc']
            else:
                self.history_vec[self.cur_conver_step] = self.history_dict['ask_single_fail']

        elif conv_action == 1: 
            
            # predict recommended items in slots of interest
            output = self.output_dict['item_policy']
            cur_slot_results = [output[:, s].argmax(dim=-1).item() for s in self.output_dict['slots']]

            # collect item_policy info
            mode = 'item_policy'
            if mode in self.side_policies:
                for s in self.output_dict['slots']:

                    # get obs and values
                    embed = self.output_dict['embed'][:, s]
                    obs = self._get_state(embed=embed, backbone=False)
                    with torch.no_grad():
                        _, values, _ = self.side_policies[mode](obs_as_tensor(obs, 'cuda'))

                    # get actions and log_probs
                    model_output = self.output_dict[mode][:, s]
                    actions = model_output.argmax(dim=-1)
                    log_probs = model_output[:, actions.item()]
                    
                    r = float(actions.item() in (self.target_set - set(self.partial_list)))
                    self.side_records[mode].append({'obs': obs, 'reward': r, 'action': actions.cpu(), 'value': values.flatten().cpu(), 'log_prob': log_probs.cpu()})

            # record and update
            self.delta_record[mode].extend(cur_slot_results)

            self.all_set = set(cur_slot_results) | self.all_set - {self.bos, self.eos, self.pad}

            partial_delta = [r for r in cur_slot_results if r in (self.target_set - set(self.partial_list))]
            self.partial_list = self.partial_list + partial_delta
            
            for i in range(len(self.black_lists)):
                self.black_lists[i] = list(set(self.black_lists[i]) | self.all_set | set(self.partial_list)) # existing black list U all recommmend set U accepted list

            if len(self.partial_list) == len(self.target_set):
                reward = metrics[self.args.reward](self.target_set, self.partial_list, self.all_set)
                self.history_vec[self.cur_conver_step] = self.history_dict['rec_bundle_suc']
                done = 1
            elif len(partial_delta) > 0:
                self.history_vec[self.cur_conver_step] = self.history_dict['rec_single_suc']
            else:
                self.history_vec[self.cur_conver_step] = self.history_dict['rec_single_fail']

            # update slots and cur
            for s, r in zip(self.output_dict['slots'], cur_slot_results):
                if r in set(partial_delta):
                    self.cur[s] = r 
                    self.slots.remove(s)
                    if len(self.blank_slots):
                        self.slots.append(self.blank_slots.pop(0))

        # record when reaching the end
        if self.cur_conver_step == self.eval_max_run - 1 and done == 0:
            reward = metrics[self.args.reward](self.target_set, self.partial_list, self.all_set) 
            if self.cur_conver_step < self.eval_max_run:
                self.history_vec[self.cur_conver_step] = self.history_dict['until_max']
            done = 1

        self.cur_conver_step += 1

        conv_record, side_records = self._get_record(conv_state, reward, conv_action, conv_value, conv_log_prob, is_done=done)

        return self._get_state(), conv_record, side_records

    def _get_record(self, state, reward, action, value, log_prob, is_done):
        conv_record = {
            'obs': state,
            'reward': reward,
            'action': torch.Tensor([action]),
            'value': value, 
            'log_prob': log_prob,
            'done': is_done
        }
        return conv_record, self.side_records

    def _update_wrap(self, done, reward):
        if done:
            res = self.wrap_info     
            for n in res:
                res[n][-1][1] = reward  # update reward, only using as the final reward
                res[n][-1][2] = done # update done
            self.wrap_info = defaultdict(list)
            return res
        else:
            return defaultdict(list)

    def eval(self):
        self.bundle_rec.eval()

    def train(self):
        self.bundle_rec.train()

    def evaluate(self, helper, trivial=0, save=None, n_samples=None):
        runs, acc, prec, rec, rewards = [], [], [], [], []
        actions, partials, test = [], [], []
        samples = range(len(self.dataset))
        if n_samples is not None:
            samples = random.sample(samples, n_samples)
        count_tqdm = tqdm(samples)

        self.eval()
        instances = {}
        if save is not None:
            records = {}
        for index in count_tqdm:  
            rewards_, actions_, partials_ = [], [], []
            state = self.reset(index=index)  # Reset environment and record the starting state
            t = 0

            if save is not None:
                records[index] = defaultdict(list)
                records[index]['target'] = list(self.target_set)

            while True:  
                if trivial == 1:
                    action = np.random.randint(0, 2) # 0 or 1
                elif trivial == 2:
                    action = 1
                elif helper is not None:
                    action, _ = helper.policy.predict(state, deterministic=True)
                    action = action.item()

                next_state, conv_record, _ = self.step(action)

                rewards_.append(conv_record['reward'])
                actions_.append(action)
                partials_.append(self.partial_list)

                state = next_state

                if save is not None:
                    if self.cur_conver_step == 1:
                        test.append(tuple(self.delta_record['item_policy']))
                    records[index]['pred_set'].append(self.delta_record)

                    if len(test) % 1000 == 0:
                        from collections import Counter 
                        print(Counter(test))

                if conv_record['done']:
                    next_state = None
                    acc.append(accuracy_score(self.target_set, self.partial_list, self.all_set))
                    prec.append(precision_score(self.target_set, self.partial_list, self.all_set))
                    rec.append(recall_score(self.target_set, self.partial_list, self.all_set))
                    runs.append(self.cur_conver_step)
                    rewards.append(np.sum(rewards_))

                    # record
                    instances[self.user] = {
                        'all_set': list(self.all_set),
                        'target_set': list(self.target_set),
                        'partial_set': self.partial_list
                    }
                    break
                t += 1

            actions.append(str(actions_))
            partials.append(partials_)
            count_tqdm.set_description(
                "prec: %.4f, rec: %.4f, acc: %.4f, run: %.4f, reward: %.4f" % (np.mean(prec), np.mean(rec), np.mean(acc), np.mean(runs), np.mean(rewards)))

        if save is not None:
            records.update({'actions': actions})
            json.dump(records, open(f"{save}.json", "w"))

        self.train()
        
        return {'prec': np.mean(prec), 'rec': np.mean(rec), 'acc': np.mean(acc), 'runs': np.mean(runs), 'rewards': np.mean(rewards)}