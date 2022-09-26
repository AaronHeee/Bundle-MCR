import os 
import json 

from collections import Counter 

path = "./"

id2item = json.load(open("item_id_lookup.json", "r"))
item2id = {id2item[i]:i for i in id2item}

all_items = json.load(open("all_items.json", "r"))
itemid2attr = {item2id[i]: [j.replace('-', ' ') for j in all_items[i][-1]] for i in all_items if i in item2id}

# item_meta = {item2id[eval(l)['id']]: eval(l)['attrs'] for l in open("steam_games.json", "r") if eval(l)['id'] in item2id}

attrs = []
for i in itemid2attr:
    attrs.extend(itemid2attr[i])
attrs_cnt = Counter(attrs)
attrs = [attr for attr, cnt in attrs_cnt.most_common() if cnt >= 10]
attrs = list(set(attrs))

attr2id = dict(zip(list(attrs), range(len(attrs))))
itemid2attrid = {i:[attr2id[j] for j in itemid2attr[i] if j in attr2id] if i in itemid2attr else [] for i in id2item}

json.dump(attr2id, open(os.path.join(path, "attr_id_lookup.json"), "w"))

json.dump(itemid2attrid, open("item_attr.json", "w"))
print(f"#attr: {len(attr2id)}")