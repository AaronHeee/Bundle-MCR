import json
import os
import re
import sys

from collections import defaultdict

import numpy as np

DIR = sys.argv[1]
METRICS = ['epoch', 'ori_prec', 'ori_rec', 'ori_acc', 'prec', 'rec', 'acc', 'f1']
# METRICS = ['epoch', 'precision', 'recall', 'accuracy']

TARGET = "test.log" if len(sys.argv) <= 2 else sys.argv[2]

# ------------------
# Collect Results
# ------------------

results = {}

for path in [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(DIR)) for f in fn]:
    if TARGET in path:
        try:
            test = json.load(open(path, 'r'))
            results[path.replace(DIR, '').replace(TARGET, '')] = {M: test[M] for M in METRICS if M in test}
        except:
            pass
        r = results[path.replace(DIR, '').replace(TARGET, '')]['rec']
        p = results[path.replace(DIR, '').replace(TARGET, '')]['prec']
        results[path.replace(DIR, '').replace(TARGET, '')].update({"f1": 2 * r * p / (r + p)})

METRICS = set([M for M in METRICS if M in test]) | set(['f1'])

L = max([len(r) for r in results]) + 5

# -----------
# Original
# -----------

print("\n[ ORIGINAL RESULTS ]\n")

print(f"{'model,': <{L}}" + ",".join(f"{m: >10}" for m in  METRICS))

for r in results:
    print(f"{r+',': <{L}}" + ','.join([f"{results[r][M]: >10.4f}" for M in METRICS]))

# --------------
# GroupBy Seed
# --------------

results_by_seed = defaultdict(dict)
for s, r in enumerate(results):
    r_ = r.rsplit('seed_', 1)
    try:
        r_, seed_ = r_
        seed_ = int(seed_.split('_')[0])
    except:
        r_ = r
        seed_ = s
    results_by_seed[r_][seed_] = results[r] 

for r in results_by_seed:
    temp_dict = {M:[] for M in METRICS}
    for s in results_by_seed[r]:
        for k in results_by_seed[r][s]:
            temp_dict[k].append(results_by_seed[r][s][k])
    results_by_seed[r]['all'] = temp_dict

L = max([len(r) for r in results_by_seed]) + 5

print("\n[ SUMMARIZED RESULTS ]\n")

print(f"{'model,': <{L}}" + ",".join(f"{m: >10},{'±'+m: >10}" for m in  METRICS))
for r in results_by_seed:
    print(f"{r+',': <{L}}" + ','.join([f"{np.mean(results_by_seed[r]['all'][M]): >10.4f},{np.std(results_by_seed[r]['all'][M]): >10.4f}" for M in METRICS]))

