#python scripts/split.py
import json
import random
from pathlib import Path

data_path = Path("/DATA/ai20resch11003/my_halloc/data/data.json")
train_path = Path("/DATA/ai20resch11003/my_halloc/data/train.json")
val_path = Path("/DATA/ai20resch11003/my_halloc/data/val.json")

# Seed for reproducibility
random.seed(42)

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

random.shuffle(data)

n_total = len(data)
n_train = int(0.9 * n_total)
n_val = n_total - n_train

train_data = data[:n_train]
val_data = data[n_train:]

assert len(set(d['id'] for d in train_data).intersection(d['id'] for d in val_data)) == 0, "Overlap detected!"

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(val_path, "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

print(f"Total rows: {n_total}, Train: {len(train_data)}, Val: {len(val_data)}")