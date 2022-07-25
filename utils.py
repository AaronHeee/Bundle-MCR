import torch
import random
import numpy as np

def random_seed(seed=None):
    if seed is None:
        seed = torch.initial_seed() % 2**32
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
    
def last_commit_msg():
    try:
        import re
        from subprocess import check_output
        hashed_id = check_output('git log -1 --pretty=format:"%H"'.split()).decode('utf-8').rstrip('\n').replace('\n', '').replace('\"', '')[:8]
        msg_short = '_'.join(re.sub("\s\s+", " ", check_output('git log -1 --oneline --pretty=format:"%s"'.split()).decode('utf-8').strip('\n').replace('\n', '').replace('\"', '')).split(' '))
        current_branch = check_output('git rev-parse --abbrev-ref HEAD'.split()).decode('utf-8').rstrip('\n').replace('\n', '').replace('\"', '')
        return current_branch, f"{msg_short}_{hashed_id}"
    except:
        return "", "no_commit"

def save_dependencies(ckpt):
    try:
        import os
        os.system(f"pip freeze > {ckpt}/requirements.txt")
    except:
        pass