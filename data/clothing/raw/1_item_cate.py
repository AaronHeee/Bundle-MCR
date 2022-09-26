import os 
import json 

from collections import Counter 

path = "./"

id2item = json.load(open("item_id_lookup.json", "r"))
item2id = {id2item[i]:i for i in id2item}

all_items = json.load(open("all_items.json", "r"))
itemid2cate = {item2id[i]: all_items[i][1][2:] for i in all_items if i in item2id}

# item_meta = {item2id[eval(l)['id']]: eval(l)['cates'] for l in open("steam_games.json", "r") if eval(l)['id'] in item2id}

cates = []
for i in itemid2cate:
    cates.extend(itemid2cate[i])
cates_cnt = Counter(cates)
cates = [cate for cate, cnt in cates_cnt.most_common() if cnt >= 10]
cates = list(set(cates))

cate2id = dict(zip(list(cates), range(len(cates))))
itemid2cateid = {i:[cate2id[j] for j in itemid2cate[i] if j in cate2id] if i in itemid2cate else [] for i in id2item}

print(f"item {sum([len(itemid2cateid[i]) == 0 for i in itemid2cateid]) / len(itemid2cateid)}")

json.dump(cate2id, open(os.path.join(path, "cate_id_lookup.json"), "w"))

json.dump(itemid2cateid, open("item_cate.json", "w"))
print(f"#cate: {len(cate2id)}")