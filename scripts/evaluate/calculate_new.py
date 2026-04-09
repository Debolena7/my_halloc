import time
import json
from typing import Optional

import fire
import torch
from tqdm import tqdm
from loguru import logger
from PIL import Image

from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from src.model import get_model
from src.datamodule.dataset import get_dataset


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
        return result
    return wrapper


@timed
def calculate_optimal_threshold(
    data_dir: str,
    save_filename: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda:1",
):
    logger.info("Load dataset")
    dataset = get_dataset("halloc_text").from_config({"file_paths": "/DATA/ai20resch11003/my_halloc/data/test/vlm_embeddings/text/", "all_flag": False})
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=2,
        pin_memory=False,
        shuffle=False,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info("Load model")
    model = get_model("halloc_text")(
        model_name_or_path="bert-base-uncased",
        vlm_out_dim=4096,
    )
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        target_prefix = "model."
        processed_checkpoint = {
            k.replace(target_prefix, ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith(target_prefix) and "position_ids" not in k
        }
        model.load_state_dict(processed_checkpoint)
    model.eval()
    model = model.to(device)

    logger.info("Start model inference")

    # Containers for probabilities and labels
    all_obj_probs, all_att_probs, all_rel_probs = [], [], []
    all_sce_probs, all_oth_probs, all_total_probs = [], [], []
    all_obj_labels, all_att_labels, all_rel_labels = [], [], []
    all_sce_labels, all_oth_labels, all_total_labels = [], [], []
    
    # Containers for storing predictions per row
    pred_results = []

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_masks = batch["attention_masks"].to(device)
        image_paths = batch["image_paths"]
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

        obj_labels = batch["obj_labels"].to(device)
        att_labels = batch["att_labels"].to(device)
        rel_labels = batch["rel_labels"].to(device)
        sce_labels = batch["sce_labels"].to(device)
        oth_labels = batch["oth_labels"].to(device)

        with torch.no_grad():
            obj_logits, att_logits, rel_logits, sce_logits, oth_logits = model(
                input_ids=input_ids,
                attention_masks=attention_masks,
                token_type_ids=token_type_ids,
                images=images,
                is_all=False,
            )

        attention_masks = attention_masks[:, 1:-1].reshape(-1)  # (B*L,)
        nonzero_indices = attention_masks.nonzero(as_tuple=True)

        # Flatten logits and labels for filtering
        def flatten_and_filter(logits, labels):
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            logits_filtered = logits[nonzero_indices]
            labels_filtered = labels[nonzero_indices]
            probs_filtered = torch.softmax(logits_filtered, dim=-1)
            return probs_filtered, labels_filtered

        filtered_obj_probs, filtered_obj_labels = flatten_and_filter(obj_logits, obj_labels)
        filtered_att_probs, filtered_att_labels = flatten_and_filter(att_logits, att_labels)
        filtered_rel_probs, filtered_rel_labels = flatten_and_filter(rel_logits, rel_labels)
        filtered_sce_probs, filtered_sce_labels = flatten_and_filter(sce_logits, sce_labels)
        filtered_oth_probs, filtered_oth_labels = flatten_and_filter(oth_logits, oth_labels)

        total_probs = torch.max(filtered_obj_probs[:, 1], filtered_att_probs[:, 1])
        total_probs = torch.max(total_probs, filtered_rel_probs[:, 1])
        total_probs = torch.max(total_probs, filtered_sce_probs[:, 1])
        total_probs = torch.max(total_probs, filtered_oth_probs[:, 1])
        total_probs = torch.stack([1 - total_probs, total_probs], dim=-1)
        total_labels = filtered_obj_labels | filtered_att_labels | filtered_rel_labels | filtered_sce_labels | filtered_oth_labels

        # Append all probabilities and labels for threshold search
        all_obj_probs.append(filtered_obj_probs.cpu())
        all_att_probs.append(filtered_att_probs.cpu())
        all_rel_probs.append(filtered_rel_probs.cpu())
        all_sce_probs.append(filtered_sce_probs.cpu())
        all_oth_probs.append(filtered_oth_probs.cpu())
        all_total_probs.append(total_probs.cpu())
        all_obj_labels.append(filtered_obj_labels.cpu())
        all_att_labels.append(filtered_att_labels.cpu())
        all_rel_labels.append(filtered_rel_labels.cpu())
        all_sce_labels.append(filtered_sce_labels.cpu())
        all_oth_labels.append(filtered_oth_labels.cpu())
        all_total_labels.append(total_labels.cpu())

        # Store per-row predictions (using 0.5 as temporary threshold)
        '''for i in range(len(image_paths)):
            pred_results.append({
                #"image_id": dataset[i]["id"],
                "obj_prob": filtered_obj_probs[i, 1].item(),
                "att_prob": filtered_att_probs[i, 1].item(),
                "rel_prob": filtered_rel_probs[i, 1].item(),
                "sce_prob": filtered_sce_probs[i, 1].item(),
                "oth_prob": filtered_oth_probs[i, 1].item(),
                "total_prob": total_probs[i, 1].item()
            })'''
        num_tokens = filtered_obj_probs.size(0)
        for i in range(num_tokens):
            pred_results.append({
                "obj_prob": filtered_obj_probs[i, 1].item(),
                "att_prob": filtered_att_probs[i, 1].item(),
                "rel_prob": filtered_rel_probs[i, 1].item(),
                "sce_prob": filtered_sce_probs[i, 1].item(),
                "oth_prob": filtered_oth_probs[i, 1].item(),
                "total_prob": total_probs[i, 1].item()
        })

    # Concatenate all tensors
    all_obj_probs = torch.cat(all_obj_probs).numpy()
    all_att_probs = torch.cat(all_att_probs).numpy()
    all_rel_probs = torch.cat(all_rel_probs).numpy()
    all_sce_probs = torch.cat(all_sce_probs).numpy()
    all_oth_probs = torch.cat(all_oth_probs).numpy()
    all_total_probs = torch.cat(all_total_probs).numpy()
    all_obj_labels = torch.cat(all_obj_labels).numpy()
    all_att_labels = torch.cat(all_att_labels).numpy()
    all_rel_labels = torch.cat(all_rel_labels).numpy()
    all_sce_labels = torch.cat(all_sce_labels).numpy()
    all_oth_labels = torch.cat(all_oth_labels).numpy()
    all_total_labels = torch.cat(all_total_labels).numpy()

    logger.info("Calculate optimal thresholds")
    thresholds_list = [i * 0.001 for i in range(1000)]

    best_thresholds = {}
    best_f1_scores = {}
    precision_at_best_thresholds = {}
    recall_at_best_thresholds = {}

    for task, probs, labels in tqdm(zip(
        ['obj', 'att', 'rel', 'glo', 'oth', 'total'],
        [all_obj_probs, all_att_probs, all_rel_probs, all_sce_probs, all_oth_probs, all_total_probs],
        [all_obj_labels, all_att_labels, all_rel_labels, all_sce_labels, all_oth_labels, all_total_labels]
    )):
        best_f1 = 0
        best_threshold = 0
        precision_at_best_threshold = 0
        recall_at_best_threshold = 0
        for threshold in thresholds_list:
            preds = (probs[:, 1] > threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                precision_at_best_threshold = precision
                recall_at_best_threshold = recall
        best_thresholds[task] = best_threshold
        best_f1_scores[task] = best_f1
        precision_at_best_thresholds[task] = precision_at_best_threshold
        recall_at_best_thresholds[task] = recall_at_best_threshold

    # Apply best thresholds to stored predictions
    for row in pred_results:
        row["obj_pred"] = int(row["obj_prob"] > best_thresholds["obj"])
        row["att_pred"] = int(row["att_prob"] > best_thresholds["att"])
        row["rel_pred"] = int(row["rel_prob"] > best_thresholds["rel"])
        row["sce_pred"] = int(row["sce_prob"] > best_thresholds["glo"])
        row["oth_pred"] = int(row["oth_prob"] > best_thresholds["oth"])
        row["total_pred"] = int(row["total_prob"] > best_thresholds["total"])

    # Save everything
    result = {
        "checkpoint_path": checkpoint_path,
        "thresholds": best_thresholds,
        "f1_score": best_f1_scores,
        "precision": precision_at_best_thresholds,
        "recall": recall_at_best_thresholds,
        "inference": pred_results
    }

    logger.info("Save optimal thresholds and inference results")
    with open(save_filename, "w") as f:
        json.dump(result, f, indent=4)
    logger.info(f"Saved results to {save_filename}")


if __name__ == "__main__":
    fire.Fire(calculate_optimal_threshold)