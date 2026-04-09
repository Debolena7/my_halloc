import time
import json
import torch
import fire
from typing import Optional
from tqdm import tqdm
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertTokenizer

from src.model import get_model
from src.datamodule.dataset import get_dataset

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
        return result
    return wrapper

def flatten_and_mask(logits, labels, mask):
    """
    Dynamically aligns the mask to the logit sequence length to prevent IndexErrors.
    """
    # B, L, C
    batch_size, seq_len, num_classes = logits.shape
    # Slice mask to match the actual sequence length returned by the model
    current_mask = mask[:, :seq_len].bool()
    
    # Reshape for masking
    logits_flat = logits.reshape(-1, num_classes)
    labels_flat = labels[:, :seq_len].reshape(-1)
    mask_flat = current_mask.reshape(-1)
    
    # Filter only valid (non-padded) tokens
    indices = torch.nonzero(mask_flat, as_tuple=True)
    return logits_flat[indices], labels_flat[indices]

@timed
def calculate_inference(
    save_filename: str,
    data_dir: str = None,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda:0",
    batch_size: int = 16,
):
    logger.info("Initializing environment...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Data Loading
    dataset = get_dataset("halloc_text").from_config({
        "file_paths": "/DATA/ai20resch11003/my_halloc/data/test/vlm_embeddings/text/", 
        "all_flag": False
    })
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    # Model Setup
    model = get_model("halloc_text")(
        model_name_or_path="bert-base-uncased",
        vlm_out_dim=4096,
    )
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() 
                     if k.startswith("model.") and "position_ids" not in k}
        model.load_state_dict(state_dict)
    
    model.to(device).eval()

    all_results = []
    final_labels = []
    final_preds = []

    logger.info(f"Running inference on {len(dataset)} samples...")
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_masks"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        img_paths = batch["image_paths"]
        images = [Image.open(p).convert("RGB") for p in img_paths]

        # Task Ground Truths
        task_names = ["obj", "att", "rel", "sce", "oth"]
        gts = [batch[f"{t}_labels"].to(device) for t in task_names]

        with torch.no_grad():
            # Model returns a tuple of 5 logits
            logits_tuple = model(
                input_ids=input_ids,
                attention_masks=attn_mask,
                token_type_ids=token_type_ids,
                images=images,
                is_all=False,
            )

        # 1. Align and Flatten for Batch Metrics
        batch_task_preds = []
        batch_task_gts = []
        
        # We use a base mask starting from index 1 to skip [CLS]
        base_mask = attn_mask[:, 1:] 

        for i in range(5):
            f_logits, f_gts = flatten_and_mask(logits_tuple[i], gts[i], base_mask)
            batch_task_preds.append(f_logits.argmax(dim=-1).cpu())
            batch_task_gts.append(f_gts.cpu())

        # Stack tasks: (Tokens_in_batch, 5)
        batch_preds_matrix = torch.stack(batch_task_preds, dim=1)
        batch_gts_matrix = torch.stack(batch_task_gts, dim=1)
        
        final_preds.append(batch_preds_matrix)
        final_labels.append(batch_gts_matrix)

        # 2. Detailed Per-Image Analysis for JSON
        # We iterate through the batch to map tokens back to words
        for b_idx in range(len(img_paths)):
            # Determine actual sequence length for this specific image
            seq_len = logits_tuple[0].size(1)
            sample_mask = base_mask[b_idx, :seq_len].bool()
            
            # Decode tokens
            raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[b_idx, 1:])
            valid_tokens = [raw_tokens[j] for j in range(len(raw_tokens)) if j < seq_len and sample_mask[j]]
            
            # Get predictions for this specific image's valid tokens
            sample_analysis = []
            for t_idx, token in enumerate(valid_tokens):
                # logits_tuple[task][batch_item][token_index]
                token_preds = {task_names[i]: int(logits_tuple[i][b_idx, t_idx].argmax()) for i in range(5)}
                sample_analysis.append({
                    "word": token,
                    "hallucinations": token_preds
                })

            all_results.append({
                "image": img_paths[b_idx],
                "analysis": sample_analysis
            })

    # --- Global Metric Calculation ---
    full_preds = torch.cat(final_preds).numpy()
    full_gts = torch.cat(final_labels).numpy()

    metrics_report = {}
    for i, task in enumerate(task_names):
        p, r, f1, _ = precision_recall_fscore_support(
            full_gts[:, i], full_preds[:, i], average="binary", zero_division=0
        )
        metrics_report[task] = {"precision": float(p), "recall": float(r), "f1": float(f1)}
        logger.info(f"{task.upper()} | F1: {f1:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")

    # --- Save Output ---
    output = {
        "config": {"checkpoint": checkpoint_path, "batch_size": batch_size},
        "overall_metrics": metrics_report,
        "predictions": all_results
    }

    with open(save_filename, "w") as f:
        json.dump(output, f, indent=4)
    
    logger.success(f"Inference complete. Results saved to {save_filename}")

if __name__ == "__main__":
    fire.Fire(calculate_inference)