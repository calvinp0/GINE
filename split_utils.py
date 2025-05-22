# split_utils.py
import os, json, torch

def get_or_make_splits(dataset, seed: int, split_path: str, ratios=(0.8, 0.1, 0.1)):
    """
    Returns dict(train, val, test) → list[int] with fixed indices.
    Creates the JSON on first call, else just loads it.
    """
    if os.path.exists(split_path):
        with open(split_path) as f:
            return json.load(f)

    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    t = int(ratios[0] * n)
    v = int(ratios[1] * n)
    splits = {
        "train": perm[:t],
        "val":   perm[t:t + v],
        "test":  perm[t + v:],
    }
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(splits, f)
    return splits
