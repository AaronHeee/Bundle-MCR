def f1_score(target_set, pred_set, all_set):
    r = recall_score(target_set, pred_set, all_set)
    p = precision_score(target_set, pred_set, all_set)
    return 2 * r * p / (r + p + 1e-8)

def recall_score(target_set, pred_set, all_set):
    return len(pred_set) / len(target_set)

def precision_score(target_set, pred_set, all_set):
    return len(pred_set) / len(all_set) if len(all_set) else 0

def accuracy_score(target_set, pred_set, all_set):
    return len(pred_set) / len(all_set | target_set)

metrics = {
    'f1': f1_score,
    'rec': recall_score,
    'prec': precision_score,
    'acc': accuracy_score
}