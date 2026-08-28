# HalLoc (adapted): Token-level Hallucination Localization

An adaptation of **HalLoc / HalLocalizer** — *"HalLoc: Token-level Hallucination
Localization for Vision-Language Models"* (Park, Kim & Kim, CVPR 2025,
[paper](https://openaccess.thecvf.com/content/CVPR2025/html/Park_HalLoc_Token-level_Localization_of_Hallucinations_for_Vision_Language_Models_CVPR_2025_paper.html),
[original code](https://github.com/dbsltm/cvpr25_halloc)) — retargeted from the
original HalLoc benchmark onto **[our LID dataset](https://huggingface.co/datasets/NLIP-lab/LID)** of [Aerial Mirage (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/papers/Basak_Aerial_Mirage_Unmasking_Hallucinations_in_Large_Vision_Language_Models_WACV_2025_paper.pdf), [GitHub link](https://github.com/Debolena7/Aerial_Mirage).

This repo trains and evaluates the hallucination detector on annotations from the LID dataset.

## Model variant used

| Variant | Config (`model=`) | Input | Notes |
|---|---|---|---|
| **Text-based** (`halloc_text`) | `src/model/halloc_text.py` | Image + **tokenized response text**, re-encoded by a frozen ResNet-50 + `bert-base-uncased` tokenizer, fed into `VisualBertModel` | This is the variant actually used in this adaptation (see `config/train.yaml` defaults) |

Both text and image embedding based variants share the same backbone: `ResNetModel` (`microsoft/resnet-50`) for image features, `VisualBertModel` (`uclanlp/visualbert-vqa-coco-pre`) as
the multimodal encoder, and five independent linear classification heads
`obj_head` (object), `att_head` (attribute), `rel_head` (relationship), `sce_head` (scene), `oth_head` (other), plus an optional `all_head` for a combined "any hallucination" label on top of the encoder's per-token output.

## Repository structure
```
my_halloc/
├── train.py # Training entry point (Hydra + PyTorch Lightning)
├── config/ # Hydra configuration files
│ ├── train.yaml # root config — composes all the below
│ ├── datamodule/ # halloc.yaml / halloc_text.yaml + shared defaults
│ ├── dataset/ # train/val dataset configs (points at vlm_embeddings dirs)
│ ├── model/ # halloc.yaml / halloc_text.yaml (backbone names, dims)
│ ├── module/ # Lightning module selector
│ ├── optimizer/ # AdamW (lr=1e-5)
│ ├── scheduler/ # cosine / multi_step LR schedules
│ ├── loss/ # cross-entropy (per-head, weight=[1,1])
│ ├── metric/ # accuracy, AUROC, precision/recall @ threshold=0.01
│ ├── callback/ # ModelCheckpoint (monitors obj/valid_auroc) + LR monitor
│ ├── trainer/ # PL Trainer settings (2 GPUs, DDP, 5 epochs by default)
│ └── experiment/ # work_dir / run name / seed
│
├── src/ # Core library
│ ├── model/
│ │ ├── halloc.py # embedding-based HallocModel
│ │ └── halloc_text.py # text-based HallocTextModel (used in this adaptation)
│ ├── module/ # PyTorch Lightning training/eval loops
│ │ ├── loss/ # cross-entropy loss wrapper
│ │ ├── metric/ # classification metrics (AUROC, precision/recall)
│ │ └── optimizer/ # optimizer + LR scheduler construction
│ ├── datamodule/ # LightningDataModules + dataset classes
│ │ └── dataset/
│ │ ├── halloc.py # loads pre-extracted VLM embedding .npy files
│ │ └── halloc_text.py # loads tokenized input_ids/attention_masks/labels .npy files
│ ├── message/ # inter-component messaging helpers
│ └── utils/ # logging / misc utilities
│
├── scripts/ # Pipeline scripts (see "Pipeline" below)
│ ├── extract/ # Step 2: extract VLM hidden states / tokenize text
│ ├── postprocess/ # Step 1: align char-level hallucination spans to tokens
│ ├── evaluate/ # Step 4: threshold search + evaluation
│ ├── calibration/ # Step 5: ECE / ACE calibration analysis
│ ├── map_tokens.py # Post-eval: collapse token-level obj predictions into words
│ ├── merge_labels.py # Post-eval: attach ground-truth object labels to predictions
│ ├── split.py # Train/val split (90/10, seed=42)
│ └── prf1.py # Word-level precision/recall/F1 (lemmatized, multi-label)
│
├── outputs/halloc_text/ # Example run outputs (checked in for reference)
│ ├── results3.json / results_old.json # Raw per-token predictions + overall_metrics from evaluate scripts
│ ├── mapped_results.jsonl # Output of map_tokens.py
│ └── results-fin.jsonl # Output of merge_labels.py (predictions + GT labels merged)
│
├── environment.yml # Conda environment (name: halloc)
└── data/
```

## Data

The pipeline expects a `data/` directory with the following:
```
data/
├── data.json # Full annotated set (input to scripts/split.py)
├── train.json / val.json / test.json # HalLoc-format samples (id, image_id, prompt,
│ # hallucinated_text, annotations{object,attribute,...})
├── {split}_text_postprocessed.json # Output of scripts/postprocess/postprocess_text.py
├── {split}_internvl_postprocessed.json # Same, for the InternVL embedding pipeline
└── {split}/vlm_embeddings/{text,internvl,...}/
├── input_ids.npy / attention_masks.npy / token_type_ids.npy (text variant)
├── image_paths.npy
└── obj_labels.npy / att_labels.npy / rel_labels.npy / sce_labels.npy / oth_labels.npy
```

Each raw sample follows the original HalLoc schema:

```json
{
  "id": "sample_id",
  "image_id": "0000360_06861_d_0000748.jpg",
  "prompt": "<image> Describe the image briefly.",
  "hallucinated_text": "The image is an aerial view of ...",
  "annotations": {
    "object": [{"obj": {"name": "truck", "char_index": "51:56"}}]
  }
}
```

## Setup

```bash
conda env create -f environment.yml
conda activate halloc
```

Requirements: Python ≥3.8, PyTorch ≥1.13, PyTorch Lightning, Hydra
(`hydra-core`, `hydra-colorlog`), HuggingFace `transformers` + `accelerate`,
`fire`, `loguru`, `scikit-learn`, `nltk`, `Pillow`.


Update the paths in the python files before running, for e.g., `data_path`, `image_dir`, `save_dir`, `file_paths`, `checkpoint_path`, etc

## Pipeline

### Step 0: We do the Train/val split first.

```bash
python scripts/split.py
```
Shuffles `data/data.json` (seed 42) into a 90-10 `train.json` / `val.json`
split with no `id` overlap.

### Step 1: We postprocess annotations

Aligns the character-level hallucination spans in `annotations` to
tokenizer-specific token indices (BERT tokenizer, 512-token max length),
producing a `tokenized_text` field and per-annotation `token_index`:

```bash
python scripts/postprocess/postprocess_text.py --input_path data/train.json
python scripts/postprocess/postprocess_text.py --input_path data/val.json
python scripts/postprocess/postprocess_text.py --input_path data/test.json
```
(writes `data/{split}_text_postprocessed.json`)

### Step 2: We extract VLM embeddings / tokenized inputs

For the **text-based** variant, tokenizes the (image, response) pairs and
saves `input_ids` / `attention_masks` / `token_type_ids` / per-category
token-level labels as `.npy` arrays:

```bash
PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=1 \
accelerate launch --num_processes=1 scripts/extract/extract_vlm_embeddings_text.py \
    --data_path data/train_text_postprocessed.json \
    --image_dir /DATA/ai20resch11003/all_images/ \
    --save_dir data/train/vlm_embeddings/text \
    --batch_size 8
# we repeat this command for val_text_postprocessed.json / test_text_postprocessed.json
```

For the **embedding-based** variant, you can run the model-specific extractor (LLaVA
/ InternVL / InstructBLIP / MiniGPT-4) to get the hidden-state embeddings
instead of raw tokens:

```bash
PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=0 \
accelerate launch --num_processes=1 scripts/extract/extract_vlm_embeddings_internvl.py \
    --data_path data/train_internvl_postprocessed.json \
    --image_dir /DATA/ai20resch11003/all_images/ \
    --save_dir data/train/vlm_embeddings/internvl \
    --batch_size 8
# repeat for val / test, and for _llava / _iblip / _minigpt4 variants as needed
```

### Step 3: Train HalLocalizer

Text-based variant (used by our adaptation — matches `config/train.yaml`'s
defaults: `datamodule=halloc_text`, `module=halloc_text`, `model=halloc_text`):

```bash
PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=1,2 \
python train.py \
    datamodule=halloc_text \
    module=halloc_text \
    model=halloc_text \
    experiment.name=halloc_text
```

For Embedding-based variant, you can use:

```bash
PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=0,1 \
python train.py \
    experiment.name=halloc \
    experiment.work_dir=./outputs
```


Checkpoints and logs are generated in `${experiment.work_dir}/${experiment.name}/logs`.

### Step 4a: We Evaluate

Find per-category decision thresholds on the validation split, then run full
evaluation on the checkpoint:

```bash
export PYTHONPATH=$(pwd)
PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=1,2 \
python scripts/evaluate/calculate_optimal_threshold_text.py \
    --checkpoint_path logs/halloc_text/version_0/checkpoints/best.ckpt \
    --data_dir data/val/vlm_embeddings/text \
    --save_filename outputs/halloc_text/results.json

PYTORCH_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" CUDA_VISIBLE_DEVICES=1,2 \
python scripts/evaluate/calculate_new_3.py \
    --checkpoint_path logs/halloc_text/version_0/checkpoints/best.ckpt \
    --data_dir data/val/vlm_embeddings/text \
    --save_filename outputs/halloc_text/results-new.json
```

- `calculate_optimal_threshold.py` / `_text.py` — sweeps decision thresholds
  per hallucination category on the validation set to maximize a target
  metric, saving `evaluation/thresholds/thresholds.json`.
- `evaluate_single.py` / `evaluate_single_text.py` — runs the trained model
  over a held-out split at the chosen thresholds and reports per-category
  precision/recall/F1/AUROC.
- `calculate_new.py` / `_2.py` / `_3.py` — later iterations of the evaluation
  script (in this repo, `results3.json` / `results_old.json` under
  `outputs/halloc_text/` are examples of their output: a `config` block, an
  `overall_metrics` block, and a `predictions` list of
  `{"image": ..., "analysis": [{"word": ..., "hallucinations": {"obj":0/1, "att":..., "rel":..., "sce":..., "oth":...}}, ...]}`
  — i.e. one row per response token with a binary flag per hallucination
  category).

### Step 4b: Word-level post-processing

The raw per-token `results*.json` output is turned into a word-level,
lemmatized evaluation via three helper scripts:

```bash
python scripts/map_tokens.py     # results3.json -> mapped_results.jsonl
                                  #   merges BERT subword ("##") tokens back into
                                  #   whole words, keeps only words flagged as
                                  #   object-hallucinated, dedupes -> "response" field

python scripts/merge_labels.py   # mapped_results.jsonl + test.json -> results-fin.jsonl
                                  #   attaches the ground-truth object-hallucination
                                  #   labels for each sample as a "labels" field

python scripts/prf1.py           # results-fin.jsonl -> classification report
                                  #   lemmatizes + multi-label-binarizes both
                                  #   "labels" and "response", then reports
                                  #   precision/recall/F1 per class via sklearn
```


## Acknowledgements

Our implementation is adapted from [HalLoc](https://github.com/dbsltm/cvpr25_halloc) (Park et al., CVPR 2025). We thank the authors for releasing their code.


## Citation

If you use this code adaptation, please cite our paper (citation will be updated soon) and the original HalLoc paper:

```bibtex
@inproceedings{park2025halloc,
  title={HalLoc: Token-level Hallucination Localization for Vision-Language Models},
  author={Park, Eunkyu and Kim, Minyeong and Kim, Gunhee},
  booktitle={CVPR},
  year={2025}
}
```