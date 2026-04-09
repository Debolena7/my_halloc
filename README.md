# HalLoc: Token-level Hallucination Localization for Vision-Language Models

[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-blue)](https://openaccess.thecvf.com/CVPR2025)

Official codebase and dataset for the CVPR 2025 paper:
**"HalLoc: Token-level Hallucination Localization for Vision-Language Models"**
[Eunkyu Park\*, Minyeong Kim\*, Gunhee Kim](https://vision.snu.ac.kr), Seoul National University
(\* equal contribution)

[Paper PDF](https://arxiv.org/abs/2506.10286) | [Project Page](https://dbsltm.github.io/HalLoc.github.io/) | [HuggingFace Dataset](https://huggingface.co/datasets/uunicee/HalLoc)

---

## Overview

**HalLoc** introduces a benchmark and detection model for **token-level hallucination localization** in vision-language model (VLM) outputs.
Unlike prior works, HalLoc supports:

- **Fine-grained hallucination detection** across object, attribute, relationship, and scene categories
- **Probabilistic outputs**, enabling nuanced interpretation via calibrated confidence scores
- **Real-time detection**, designed for plug-and-play integration with VLMs

The dataset includes **155K token-level annotated samples** across Visual Question Answering, Instruction Following, and Image Captioning tasks.

We also provide a lightweight detection model, **HalLocalizer**, built on VisualBERT that operates on VLM hidden-state embeddings + images.

---

## Repository Structure

```
my_halloc/
├── train.py                               # Training entry point (Hydra + PyTorch Lightning)
├── config/                                # Hydra configuration files
│   ├── train.yaml                         #   root config
│   ├── datamodule/                        #   data loading configs
│   ├── dataset/                           #   dataset configs
│   ├── model/                             #   model architecture configs
│   ├── module/                            #   lightning module configs
│   ├── optimizer/                         #   optimizer configs (AdamW)
│   ├── scheduler/                         #   LR scheduler configs (cosine, multi-step)
│   ├── loss/                              #   loss function configs (cross-entropy)
│   ├── metric/                            #   metric configs (classification)
│   ├── callback/                          #   callback configs
│   ├── trainer/                           #   trainer configs
│   └── experiment/                        #   experiment presets
│
├── src/                                   # Core library
│   ├── model/                             #   HalLocalizer model definitions
│   │   ├── halloc.py                      #     embedding-based model (VLM hidden states)
│   │   └── halloc_text.py                 #     text-based model (tokenized input)
│   ├── module/                            #   PyTorch Lightning modules
│   │   ├── loss/                          #     loss functions
│   │   ├── metric/                        #     evaluation metrics
│   │   └── optimizer/                     #     optimizer + scheduler setup
│   ├── datamodule/                        #   data modules + datasets
│   │   └── dataset/                       #     dataset implementations
│   ├── message/                           #   inter-component messaging
│   └── utils/                             #   logging utilities
│
└── scripts/                               # Pipeline scripts
    ├── extract/                           #   Step 1: Extract VLM embeddings
    │   ├── extract_vlm_embeddings_llava.py
    │   ├── extract_vlm_embeddings_internvl.py
    │   ├── extract_vlm_embeddings_iblip.py
    │   ├── extract_vlm_embeddings_minigpt4.py
    │   ├── extract_vlm_embeddings_text.py
    │   └── extract_vlm_embeddings_text_blur.py
    │
    ├── postprocess/                       #   Step 2: Align token indices
    │   ├── postprocess_llava.py
    │   ├── postprocess_internvl.py
    │   ├── postprocess_iblip.py
    │   ├── postprocess_minigpt4.py
    │   └── postprocess_text.py
    │
    ├── evaluate/                          #   Step 4: Evaluate + find thresholds
    │   ├── evaluate_single.py
    │   ├── evaluate_single_text.py
    │   ├── calculate_optimal_threshold.py
    │   └── calculate_optimal_threshold_text.py
    │
    └── calibration/                       #   Step 5: Calibration analysis
        ├── calculate_calibration_error_halloc_ece.py
        ├── calculate_calibration_error_halloc_ace.py
        ├── calculate_calibration_error_internvl_ece.py
        ├── calculate_calibration_error_internvl_ace.py
        └── calculate_logprob_internvl.py
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.13
- PyTorch Lightning
- Hydra (`hydra-core`, `hydra-colorlog`)
- HuggingFace Transformers
- Accelerate
- `fire`, `loguru`, `scikit-learn`, `Pillow`

For VLM-specific extraction scripts, you also need the corresponding VLM libraries:
- **LLaVA**: [liuhaotian/llava-v1.5-7b](https://github.com/haotian-liu/LLaVA)
- **InternVL**: [OpenGVLab/InternVL2-8B](https://github.com/OpenGVLab/InternVL)
- **InstructBLIP / MiniGPT-4**: respective repositories

---

## Dataset

The HalLoc dataset contains 155K token-level annotated samples with hallucination labels across five categories: **object**, **attribute**, **relationship**, **scene**, and **other**.

<!-- TODO: Add download link -->
Download link: _Coming soon (HuggingFace / Google Drive)_

The annotation JSON files follow this structure:
```json
{
  "id": "sample_id",
  "image_id": "image_id",
  "prompt": "<image> question text",
  "hallucinated_text": "model response text",
  "tokenized_text": ["token1", "token2", ...],
  "annotations": {
    "object": [{"entity": {"name": "...", "char_index": "start:end", "token_index": "start:end"}}],
    "attribute": [...],
    "relationship": [...],
    "scene": [...],
    "other": [...]
  }
}
```

Images come from [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/) (VG_100K / VG_100K_2) and [MS-COCO](https://cocodataset.org/) (train2014 / val2014).

---

## Pipeline

The full workflow has five stages:

conda activate halloc
cd my_halloc

### Step 1: Postprocess annotations

Align character-level hallucination annotations to VLM-specific tokenizations:

```bash
python scripts/postprocess/postprocess_internvl.py \
    --input_path data/test.json
```

python scripts/postprocess/postprocess_text.py \
    --input_path data/train.json

python scripts/postprocess/postprocess_text.py \
    --input_path data/val.json

python scripts/postprocess/postprocess_text.py \
    --input_path data/test.json

### Step 2: Extract VLM embeddings

Run a VLM in forward mode to extract hidden-state embeddings for each sample. Uses HuggingFace Accelerate for multi-GPU:

```bash
PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 scripts/extract/extract_vlm_embeddings_text.py \
    --data_path data/train_text_postprocessed.json \
    --image_dir /DATA/ai20resch11003/all_images/ \
    --save_dir data/train/vlm_embeddings/text \
    --batch_size 8


PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 scripts/extract/extract_vlm_embeddings_text.py \
    --data_path data/val_text_postprocessed.json \
    --image_dir /DATA/ai20resch11003/all_images/ \
    --save_dir data/val/vlm_embeddings/text \
    --batch_size 8

PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 scripts/extract/extract_vlm_embeddings_text.py \
    --data_path data/test_text_postprocessed.json \
    --image_dir /DATA/ai20resch11003/all_images/ \
    --save_dir data/test/vlm_embeddings/text \
    --batch_size 8

PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 scripts/extract/extract_vlm_embeddings_internvl.py \
    --data_path data/train_internvl_postprocessed.json \
    --image_dir /DATA/ai20resch11003/all_images/ \
    --save_dir data/train/vlm_embeddings/internvl \
    --batch_size 8

PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 scripts/extract/extract_vlm_embeddings_internvl.py \
    --data_path data/val_internvl_postprocessed.json \
    --image_dir /DATA/ai20resch11003/all_images/ \
    --save_dir data/val/vlm_embeddings/internvl \
    --batch_size 8

PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 scripts/extract/extract_vlm_embeddings_internvl.py \
    --data_path data/test_internvl_postprocessed.json \
    --image_dir /DATA/ai20resch11003/all_images/ \
    --save_dir data/test/vlm_embeddings/internvl \
    --batch_size 8
```
### Step 3: Train HalLocalizer

Train the hallucination detection model using Hydra configs:

```bash
PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=0,1 python train.py \
    experiment.name=halloc \
    experiment.work_dir=./outputs
```

To use the text-based model variant:

```bash
PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=1,2 python train.py \
    datamodule=halloc_text \
    module=halloc_text \
    model=halloc_text \
    experiment.name=halloc_text
```

Override any config value via the command line (Hydra syntax). See `config/` for all options.

### Step 4: Evaluate

Find optimal per-category thresholds on validation set, then evaluate:

```bash
python scripts/evaluate/calculate_optimal_threshold.py \
    --checkpoint_path outputs/halloc_internvl/best.ckpt \
    --data_dir data/val/vlm_embeddings/internvl

export PYTHONPATH=$(pwd)
PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=1,2 python scripts/evaluate/calculate_optimal_threshold_text.py \
    --checkpoint_path logs/halloc_text/version_0/checkpoints/best.ckpt \
    --data_dir data/val/vlm_embeddings/text \
    --save_filename outputs/halloc_text/results.json

export PYTHONPATH=$(pwd)
PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=1,2 python scripts/evaluate/calculate_new_3.py \
    --checkpoint_path logs/halloc_text/version_0/checkpoints/best.ckpt \
    --data_dir data/val/vlm_embeddings/text \
    --save_filename outputs/halloc_text/results-new.json

python scripts/evaluate/evaluate_single.py \
    --threshold_path evaluation/thresholds/thresholds.json \
    --checkpoint_path outputs/halloc_internvl/best.ckpt \
    --data_dir data/val/vlm_embeddings/internvl

```

note: subwords not explicitly considered while evaluating the inference results.

### Step 5: Calibration analysis (optional)

Compute Expected Calibration Error (ECE) and Adaptive Calibration Error (ACE) with temperature scaling:

```bash
python scripts/calibration/calculate_calibration_error_halloc_ece.py --subset vqa
python scripts/calibration/calculate_calibration_error_internvl_ece.py --subset vqa
```
---

## Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{park2025halloc,
  title={HalLoc: Token-level Hallucination Localization for Vision-Language Models},
  author={Park, Eunkyu and Kim, Minyeong and Kim, Gunhee},
  booktitle={CVPR},
  year={2025}
}
```

## Acknowledgements

This work was supported by Seoul National University and grants from MSIT/IITP (South Korea).

---

## Contact

For questions or collaborations, feel free to open an issue or email us at:
eunkyu.park@vision.snu.ac.kr
