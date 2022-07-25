import os 
import json
import random
import torch

from collections import defaultdict

class Helper(object):
    def __init__(self, path, mode='cate'):
        self.item2tagid = json.load(open(os.path.join(path, "raw", f"item_{mode}.json"), "r"))
        self.item2tagid = {int(item): self.item2tagid[item] for item in self.item2tagid}
        self.tag2id = json.load(open(os.path.join(path, "raw", f"{mode}_id_lookup.json"), "r"))
        self.tag2id = {t: int(self.tag2id[t]) for t in self.tag2id}
        self.pad = max([self.tag2id[i] for i in self.tag2id]) + 1

        self.tagid2item = defaultdict(set)
        for i in self.item2tagid:
            for t in self.item2tagid[i]:
                self.tagid2item[t].add(i)

    def reset(self):
        self.pos_tag = defaultdict(set)
        self.neg_tag = defaultdict(set)
        self.pos_tag_per_slot = defaultdict(set) 
        self.neg_tag_per_slot = defaultdict(set) 

    def select(self, k=3, slot=None):
        cands = list(set(self.tagid2item.keys()) - self.pos_tag_per_slot[slot] - self.neg_tag_per_slot[slot])
        tagids = random.sample(cands, k=min(k, len(cands)))
        return tagids 

    def record(self, tag, mode='pos', slot=None):
        if mode == 'pos':
            self.pos_tag_per_slot[slot].add(tag)
        elif mode == 'neg':
            for s in self.neg_tag_per_slot:
                self.neg_tag_per_slot[s].add(tag)

    def summarize(self, k):
        pos_tensor, neg_tensor = 0, 0

        if len(self.pos_tag_per_slot):
            # max_tag_num = max([len(self.pos_tag_per_slot[i]) for i in self.pos_tag_per_slot])
            max_tag_num = 1
            pos_tensor = torch.full((k, 1, max_tag_num), fill_value=self.pad).long()
            # pos_tensor = torch.full((k, 1), fill_value=self.pad).long()
            for i in self.pos_tag_per_slot:
                tags = torch.LongTensor(list(self.pos_tag_per_slot[i])[:max_tag_num])
                pos_tensor[i, :, :len(tags)] = tags
            pos_tensor = pos_tensor.cuda()

        if len(self.neg_tag_per_slot):
            # max_tag_num = max([len(self.neg_tag_per_slot[i]) for i in self.neg_tag_per_slot])
            max_tag_num = 1
            neg_tensor = torch.full((k, 1, max_tag_num), fill_value=self.pad).long()
            # neg_tensor = torch.full((k, 1), fill_value=self.pad).long()
            for i in self.neg_tag_per_slot:
                tags = torch.LongTensor(list(self.neg_tag_per_slot[i])[:max_tag_num])
                neg_tensor[i, :, :len(tags)] = tags
            neg_tensor = neg_tensor.cuda()

        return pos_tensor, neg_tensor