import os
import json 
import sys 
import numpy as np
import random

root_data = "./"
store_data = "../processed"

random.seed(5583)

# Data Loading

all_items = json.load(open(os.path.join(root_data, "all_items.json")))

user_bundle = json.load(open(os.path.join(root_data, "user_bundle_map.json")))
user_bundle = {int(i): user_bundle[i] for i in user_bundle}

bundle_item = json.load(open(os.path.join(root_data, "bundle_item_map.json")))
bundle_item = {int(i): bundle_item[i] for i in bundle_item}

# Data Statistics

print("user num:",  len(user_bundle))
print("item num:",  len(all_items))

print("avg bundles per user:",  np.mean([len(user_bundle[i]) for i in user_bundle]))
print("less than 4 bundles:",  (np.array([len(user_bundle[i]) for i in user_bundle]) < 4).sum() / len(user_bundle))

# Data Splitting

# Step 1. clean the user_bundle 

user_bundle_clean = {}
for u in user_bundle:
    bundles = [bundle_item[b] for b in user_bundle[u] if len(bundle_item[b]) > 1]
    if len(bundles) > 1:
        random.shuffle(bundles)  # temporal info is not important here 
        user_bundle_clean[u] = bundles

# Step 2. split offline data and leave-one-out data

user_bundle_offline = {}
user_bundle_leave_one = {} 

for u in user_bundle_clean:
    user_bundle_offline[u] = user_bundle_clean[u][:-1]
    user_bundle_leave_one[u] = user_bundle_clean[u][-1:]

# Step 3. split the online data into train valid and test by 6 : 2 : 2 

user_bundle_online = {}
user_bundle_valid = {}
user_bundle_test = {}

users = list(user_bundle_leave_one.keys())
random.shuffle(users)

users_train = set(users[: int(0.6*len(users))])
users_valid = set(users[int(0.6*len(users)):int(0.8*len(users))])
users_test = set(users[int(0.8*len(users)):])

for u in user_bundle_leave_one:

    if u in users_train :
        user_bundle_online[u] = user_bundle_leave_one[u]
    elif u in users_valid:
        user_bundle_valid[u] = user_bundle_leave_one[u] 
    else:
        user_bundle_test[u] = user_bundle_leave_one[u]
        
print(len(user_bundle_offline), len(user_bundle_online), len(user_bundle_valid), len(user_bundle_test))

# Step 4. store the data

if not os.path.exists(store_data):
    os.makedirs(store_data)

json.dump(user_bundle_offline, open(os.path.join(store_data, "user_bundle_offline.json"), "w"))
json.dump(user_bundle_online, open(os.path.join(store_data, "user_bundle_online.json"), "w"))
json.dump(user_bundle_valid, open(os.path.join(store_data, "user_bundle_valid.json"), "w"))
json.dump(user_bundle_test, open(os.path.join(store_data, "user_bundle_test.json"), "w"))
