import math
from numpy.lib.arraysetops import isin
import torch
import random
from torch import nn
import torch.nn.functional as F

INF = 1000000
MIN = 1e-6

#####################
# TRM Encoder
#####################


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        # embed
        self.embed = nn.Embedding(
            args.item_large, args.embed_size, padding_idx=args.pad)
        # low level
        self.low_trm = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=args.embed_size, nhead=args.n_heads), num_layers=1)
        # high level
        self.high_trm = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=args.embed_size, nhead=args.n_heads), num_layers=args.n_layers)

    def forward(self, src):
        # embedding: (length, batch_size, bundle_size, embed_size)
        embedded = self.embed(src) * math.sqrt(self.args.embed_size)
        length, batch_size, bundle_size, embed_size = embedded.size()

        # low level: low trm and avg pooling to (length, batch, embed_size) 
        embedded = torch.permute(embedded, (2, 0, 1, 3)).view(bundle_size, -1, embed_size) # (bundle_size, length * batch_size, embed_size)
        low_src = torch.permute(src, (2, 0, 1)) # (bundle_size, length * batch_size)
        output = self.low_trm(embedded, src_key_padding_mask=(low_src.view(bundle_size, -1) == self.args.pad).transpose(0, 1)) # (bundle_size, length*batch_size, embed_size)
        output = output.view(bundle_size, length, batch_size, embed_size)
        output = output[1:].sum(dim=0) / ((low_src[1:] != self.args.pad).sum(dim=0).unsqueeze(dim=-1) + MIN) # (length, batch_size, embed_size), discard [CLS] token embedding

        # high level: (length, batch, embed_size)
        high_src = src[:,:,0] # src[:,:,0] is item tokens or PAD tokens
        output = self.high_trm(output, src_key_padding_mask=(high_src == self.args.pad).transpose(0, 1))

        return output

#####################
# TRM Decoder
#####################


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.embed = nn.Embedding(
            args.item_large, args.embed_size, padding_idx=args.pad)
        self.trms = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=args.embed_size, nhead=2, dropout=args.dropout, dim_feedforward=args.embed_size*2) for _ in range(args.n_layers)])
        self.layer_norms = nn.ModuleList(
            [nn.Sequential(nn.Linear(args.embed_size, args.embed_size), nn.LayerNorm(args.embed_size)) for _ in range(args.n_layers+1)])

    def forward(self, partial_output, encoder_output, mask, cate_emb=0, attr_emb=0, tgt_mask=None):
        embedded = self.embed(partial_output)
        for n in range(self.args.n_layers):
            embedded = self.layer_norms[n](embedded + cate_emb + attr_emb)
            embedded = self.trms[n](embedded, encoder_output, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_mask)
        output = self.layer_norms[self.args.n_layers](embedded + cate_emb + attr_emb)
        return output

####################
# Bunt (Bundle Bert)
####################


class Bunt(nn.Module):
    def __init__(self, encoder, decoder, args):
        super(Bunt, self).__init__()
        self.args = args

        self.encoder = encoder
        self.decoder = decoder

        self.cate_pos_emb = nn.Embedding(
            args.cate_pad+1, args.embed_size, padding_idx=-1)
        self.cate_neg_emb = nn.Embedding(
            args.cate_pad+1, args.embed_size, padding_idx=-1)
        self.attr_pos_emb = nn.Embedding(
            args.attr_pad+1, args.embed_size, padding_idx=-1)
        self.attr_neg_emb = nn.Embedding(
            args.attr_pad+1, args.embed_size, padding_idx=-1)

        self.item_predictor = nn.Sequential(
            nn.Linear(args.embed_size, 2 * args.embed_size),
            nn.ReLU(),
            nn.Linear(2 * args.embed_size, args.item_large),
        )

        self.cate_predictor = nn.Sequential(
            nn.Linear(args.embed_size, 2 * args.embed_size),
            nn.ReLU(),
            nn.Linear(2 * args.embed_size, args.cate_pad+1),
        )

        self.attr_predictor = nn.Sequential(
            nn.Linear(args.embed_size, 2 * args.embed_size),
            nn.ReLU(),
            nn.Linear(2 * args.embed_size, args.attr_pad+1),
        )

        self.conv_predictor = nn.Sequential(
            nn.Linear(args.embed_size, 2 * args.embed_size),
            nn.ReLU(),
            nn.Linear(2 * args.embed_size, 2),
        )


    def forward(self, src, par, cate_pos=0, cate_neg=0, attr_pos=0, attr_neg=0):
        encoder_output = self.encoder(src)
        cate, attr = self._parse_cate_attr(cate_pos, cate_neg, attr_pos, attr_neg)

        output = self.decoder(
            par,
            encoder_output,
            mask=(src[:,:,0] == self.args.pad).transpose(0, 1),
            tgt_mask=(par == self.args.pad).transpose(0, 1),
            cate_emb=cate,
            attr_emb=attr
        )

        cate_output = F.log_softmax(self.cate_predictor(output), dim=-1)
        attr_output = F.log_softmax(self.attr_predictor(output), dim=-1)
        conv_output = F.log_softmax(self.conv_predictor(output), dim=-1)
        output = F.log_softmax(self.item_predictor(output), dim=-1)

        return output, cate_output, attr_output, conv_output

    def generate(self, src, partial=None, delta_len=2,  cate_pos=0, cate_neg=0, attr_pos=0, attr_neg=0, out_embedding=False):

        encoder_output = self.encoder(src)
        cate, attr = self._parse_cate_attr(cate_pos, cate_neg, attr_pos, attr_neg)

        batch_size = encoder_output.size(1)

        if partial is None:
            partial = torch.full(size=(1, batch_size),
                                 fill_value=self.args.bos)

        partial_size = partial.size(0)

        outputs = torch.zeros(delta_len + partial_size,
                              batch_size, self.args.item_large).cuda()
        cate_outputs = torch.zeros(delta_len + partial_size, 
                              batch_size, self.args.cate_pad+1).cuda()
        attr_outputs = torch.zeros(delta_len + partial_size, 
                              batch_size, self.args.attr_pad+1).cuda()
        conv_outputs = torch.zeros(delta_len + partial_size, batch_size, 2).cuda()

        if out_embedding:
            embed_outputs = torch.zeros(
                delta_len + partial_size, batch_size, self.args.embed_size).cuda()

        par = torch.full((delta_len+partial_size, batch_size),self.args.pad).long().cuda()
        par[:partial_size] = partial

        for t in range(partial_size, partial_size + delta_len):
            par[t] = self.args.mask
            output = self.decoder(
                par,
                encoder_output,
                mask=(src[:,:,0] == self.args.pad).transpose(0, 1),
                tgt_mask=(par == self.args.pad).transpose(0, 1),
                cate_emb=cate,
                attr_emb=attr
            )

            if out_embedding:
                embed_outputs[t] = output[t]

            cate_output = F.log_softmax(self.cate_predictor(output), dim=-1)
            attr_output = F.log_softmax(self.attr_predictor(output), dim=-1)
            conv_output = F.log_softmax(self.conv_predictor(output), dim=-1)
            output = F.log_softmax(self.item_predictor(output), dim=-1)

            indicator = par[:t]
            indicator[indicator == self.args.eos] = self.args.bos
            output[t][range(batch_size), indicator] = float('-inf')

            par[t] = output[t].data.max(1)[1]

            outputs[t] = output[t]
            cate_outputs[t] = cate_output[t]
            attr_outputs[t] = attr_output[t]
            conv_outputs[t] = conv_output[t]

        return (outputs, cate_outputs, attr_outputs, conv_outputs, embed_outputs) if out_embedding else (outputs, cate_outputs, attr_outputs, conv_outputs)

    def propose(self, src, cur, slots,  cate_pos=0, cate_neg=0, attr_pos=0, attr_neg=0, black_lists=None):

        encoder_output = self.encoder(src)
        cate, attr = self._parse_cate_attr(cate_pos, cate_neg, attr_pos, attr_neg)
        batch_size = encoder_output.size(1)

        slot_size = cur.size(0)

        outputs = torch.zeros(slot_size, batch_size, self.args.item_large).to(cur.device)
        cate_outputs = torch.zeros(slot_size, batch_size, self.args.cate_pad+1).to(cur.device)
        attr_outputs = torch.zeros(slot_size, batch_size, self.args.attr_pad+1).to(cur.device)
        conv_outputs = torch.zeros(slot_size, batch_size, 2).to(cur.device)
        embed_outputs = torch.zeros(slot_size, batch_size, self.args.embed_size).to(cur.device)

        temp_pred = []

        for s in slots:
            cur[s] = self.args.mask
            output = self.decoder(
                cur,
                encoder_output,
                mask=(src[:,:,0] == self.args.pad).transpose(0, 1),
                tgt_mask=(cur == self.args.pad).transpose(0, 1),
                cate_emb=cate,
                attr_emb=attr
            )
            embed_outputs[s] = output[s]

            cate_output = F.log_softmax(self.cate_predictor(output), dim=-1)
            attr_output = F.log_softmax(self.attr_predictor(output), dim=-1)
            conv_output = F.log_softmax(self.conv_predictor(output), dim=-1)
            output = F.log_softmax(self.item_predictor(output), dim=-1)
            output[s][:, black_lists[s] + temp_pred] = float('-inf') # exclude black list items
            cur[s] = output[s].data.max(1)[1]
            temp_pred.append(cur[s].item())

            outputs[s] = output[s]
            cate_outputs[s] = cate_output[s]
            attr_outputs[s] = attr_output[s]
            conv_outputs[s] = conv_output[s]

        return outputs, cate_outputs, attr_outputs, conv_outputs, embed_outputs

    def _parse_cate_attr(self, cate_neg, cate_pos, attr_neg, attr_pos):
        if isinstance(cate_pos, torch.Tensor):
            cate_pos = self.cate_pos_emb(cate_pos) if len(cate_pos.size()) != 3 else self.cate_pos_emb(
                cate_pos).sum(dim=2) / ((cate_pos != self.args.cate_pad).sum(dim=2).unsqueeze(dim=-1) + MIN)
        if isinstance(cate_neg, torch.Tensor):
            cate_neg = self.cate_neg_emb(cate_neg) if len(cate_neg.size()) != 3 else self.cate_neg_emb(
                cate_neg).sum(dim=2) / ((cate_neg != self.args.cate_pad).sum(dim=2).unsqueeze(dim=-1) + MIN)
        if isinstance(attr_pos, torch.Tensor):
            attr_pos = self.attr_pos_emb(attr_pos) if len(attr_pos.size()) != 3 else self.attr_pos_emb(
                attr_pos).sum(dim=2) / ((attr_pos != self.args.attr_pad).sum(dim=2).unsqueeze(dim=-1) + MIN)
        if isinstance(attr_neg, torch.Tensor):
            attr_neg = self.attr_neg_emb(attr_neg) if len(attr_neg.size()) != 3 else self.attr_neg_emb(
                attr_neg).sum(dim=2) / ((attr_neg != self.args.attr_pad).sum(dim=2).unsqueeze(dim=-1) + MIN)
        return cate_neg + cate_pos, attr_neg + attr_pos