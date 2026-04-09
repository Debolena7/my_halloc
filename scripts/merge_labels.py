# python scripts/merge_labels.py

import json

pred_file = "/DATA/ai20resch11003/my_halloc/outputs/halloc_text/mapped_results.jsonl"
gt_file = "/DATA/ai20resch11003/my_halloc/data/test.json"
out_file = "/DATA/ai20resch11003/my_halloc/outputs/halloc_text/results-fin.jsonl"

with open(gt_file, "r") as f:
    gt_data = json.load(f)

with open(pred_file, "r") as pf, open(out_file, "w") as out:
    
    for i, line in enumerate(pf):
        pred = json.loads(line)
        gt = gt_data[i]  # aligned row

        obj_list = gt.get("annotations", {}).get("object", [])
        
        labels = []
        for item in obj_list:
            name = item.get("obj", {}).get("name", "").strip().lower()
            if name:
                labels.append(name)

        seen = set()
        unique_labels = []
        for l in labels:
            if l not in seen:
                seen.add(l)
                unique_labels.append(l)

        label_str = ", ".join(unique_labels) if unique_labels else "None"

        pred["labels"] = label_str

        out.write(json.dumps(pred) + "\n")

print("Mapped JSONL with labels created!")