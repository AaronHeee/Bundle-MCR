import os 
import json 

path = "./"

id2item = json.load(open("item_id_lookup.json", "r"))
item2id = {id2item[i]:i for i in id2item}

item_meta = [eval(l) for l in open("steam_games.json", "r")]
itemid2genre = {item2id[i['id']]: i['genres'] for i in item_meta if 'id' in i and 'genres' in i and i['id'] in item2id}

genres = []
for i in itemid2genre:
    genres.extend(itemid2genre[i])
genres = list(set(genres))

genre2id = dict(zip(list(genres), range(len(genres))))
itemid2genreid = {i:[genre2id[j] for j in itemid2genre[i]] if i in itemid2genre else [] for i in id2item}

json.dump(genre2id, open(os.path.join(path, "cate_id_lookup.json"), "w"))

json.dump(itemid2genreid, open("item_cate.json", "w"))