import os 
import json 
import torch
import random

from collections import defaultdict, Counter, OrderedDict 
from stable_baselines3.common.utils import obs_as_tensor


class SimulatedUser(object):
    def __init__(self, path, noise=0):
        self.item2cate = json.load(open(os.path.join(path, "raw", "item_cate.json"), "r"))
        self.item2cate = {int(i): self.item2cate[i] for i in self.item2cate}
        self.item2attr = json.load(open(os.path.join(path, "raw", "item_attr.json"), "r"))
        self.item2attr = {int(i): self.item2attr[i] for i in self.item2attr}
        self.slot2item = {}

    def reset(self):
        self.slot2item = {}

    def user_feedback(self, tag, diff, mode='cate_policy', slot=None):

        item2tag = self.item2cate if mode == 'cate_policy' else self.item2attr  

        pos_set = []
        for i in diff:
            if i in item2tag:
                pos_set.extend(item2tag[i])
        pos_set = set(pos_set)

        return tag in pos_set

    def user_feedback_slot(self, tag, diff, mode='cate_policy', slot=None):
        item2tag = self.item2cate if mode == 'cate_policy' else self.item2attr 

        if slot not in self.slot2item:
            self.slot2item[slot] = set(diff)
        else:
            self.slot2item[slot] = set(diff) & self.slot2item[slot]

        diff_new = set([])
        for i in self.slot2item[slot]:
            if i in item2tag and tag in item2tag[i]:
                diff_new.add(i)
        if len(diff_new):
            for s in self.slot2item:
                if s == slot:
                    self.slot2item[s] = diff_new

        res = "none"

        diff_tags = set([])
        for d in diff:
            if d in item2tag:
                diff_tags |= set(item2tag[d])

        if len(diff_new) > 0:
            res = "pos"
        elif tag not in diff_tags:
            res = "neg"

        return res
         

