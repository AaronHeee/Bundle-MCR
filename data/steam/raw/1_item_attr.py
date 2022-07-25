import os
import json 

path = "./"

all_items = json.load(open(os.path.join(path, "all_items.json"), "r"))

tags = []
for i in all_items:
    tags.extend(i['tags'])
tag2id = dict(zip(set(tags), range(len(tags))))

# save tag2id
json.dump(tag2id, open(os.path.join(path, "attr_id_lookup.json"), "w"))

item2appid = json.load(open(os.path.join(path, "item_id_lookup.json"), "r"))
appid2item = {item2appid[i]: i for i in item2appid}

item2tag = {appid2item[i['appid']]: [tag2id[t] for t in i['tags']] for i in all_items if i['appid'] in appid2item}

# save item2tag
json.dump(item2tag, open("item_attr.json", "w"))

