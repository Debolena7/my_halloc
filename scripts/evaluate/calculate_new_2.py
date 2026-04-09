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
def calculate_inference(
    data_dir: str,
    save_filename: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda:0",
    batch_size: int = 16,
):
    logger.info("Loading dataset...")
    dataset = get_dataset("halloc_text").from_config({"file_paths": "/DATA/ai20resch11003/my_halloc/data/test/vlm_embeddings/text/", "all_flag": False})
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info("Loading model...")
    model = get_model("halloc_text")(
        model_name_or_path="bert-base-uncased",
        vlm_out_dim=4096,
    )
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        prefix = "model."
        processed_ckpt = {
            k.replace(prefix, ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith(prefix) and "position_ids" not in k
        }
        model.load_state_dict(processed_ckpt)
    model.to(device).eval()

    # Collect predictions
    all_results = []

    all_labels = []
    all_preds = []

    logger.info("Running inference...")
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_masks = batch["attention_masks"].to(device)
        image_paths = batch["image_paths"]
        images = [Image.open(p).convert("RGB") for p in image_paths]

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

        # Flatten logits for multi-class
        def flatten_and_mask(logits, labels):
            # (B,L,C) -> (B*L, C)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            mask = attention_masks[:, 1:-1].reshape(-1).nonzero(as_tuple=True)
            return logits[mask], labels[mask]

        obj_logits, obj_labels = flatten_and_mask(obj_logits, obj_labels)
        att_logits, att_labels = flatten_and_mask(att_logits, att_labels)
        rel_logits, rel_labels = flatten_and_mask(rel_logits, rel_labels)
        sce_logits, sce_labels = flatten_and_mask(sce_logits, sce_labels)
        oth_logits, oth_labels = flatten_and_mask(oth_logits, oth_labels)

        # Argmax for predictions
        obj_preds = obj_logits.argmax(dim=-1).cpu()
        att_preds = att_logits.argmax(dim=-1).cpu()
        rel_preds = rel_logits.argmax(dim=-1).cpu()
        sce_preds = sce_logits.argmax(dim=-1).cpu()
        oth_preds = oth_logits.argmax(dim=-1).cpu()

        # Collect labels for metrics
        all_labels.append(torch.stack([obj_labels.cpu(), att_labels.cpu(), rel_labels.cpu(),
                                       sce_labels.cpu(), oth_labels.cpu()], dim=1))
        all_preds.append(torch.stack([obj_preds, att_preds, rel_preds, sce_preds, oth_preds], dim=1))

        # Collect per-sample results
        for i in range(len(image_paths)):
            all_results.append({
                #"image_id": dataset[i]["id"],
                "obj_pred": obj_preds[i].item(),
                "att_pred": att_preds[i].item(),
                "rel_pred": rel_preds[i].item(),
                "sce_pred": sce_preds[i].item(),
                "oth_pred": oth_preds[i].item(),
            })

    # Concatenate all labels and preds
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # Compute metrics per task
    tasks = ["obj"] #"att", "rel", "sce", "oth"]
    metrics = {}
    for i, task in enumerate(tasks):
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels[:, i], all_preds[:, i], average="macro", zero_division=0
        )
        metrics[task] = {"precision": precision, "recall": recall, "f1": f1}

    logger.info("Saving inference results...")
    with open(save_filename, "w") as f:
        json.dump({"metrics": metrics}, f, indent=4)# "predictions": all_results}, f, indent=4)

    logger.info("Done!")
    return metrics, all_results


if __name__ == "__main__":
    fire.Fire(calculate_inference)