
# Model Finetuning (Uni)

How to run k-fold finetuning with pre-generated CSV splits.

## 1) Download Dataset

Download the 100k HE Patch Dataset: <https://zenodo.org/records/1214456>

Save as in dirs `CRC-VAL-HE-7K` and `NCT-CRC-HE-100K`

## 2) Put the Uni weights in place

Download `pytorch_model.bin` from <https://huggingface.co/MahmoodLab/UNI> and save it as:

```bash
foundation_models/uni.bin
```

## 3) Install deps (uv)

Install uv: <https://docs.astral.sh/uv/getting-started/installation/>

```bash
uv sync
source .venv/bin/activate
```

## 4) Prepare splits directory

`--splits_dir` must contain:

```bash
splits-nct/
  split_config.json
  test_split.csv
  fold_0/train.csv
  fold_0/val.csv
  fold_1/train.csv
  fold_1/val.csv
  ...
```

Each CSV needs columns:

* `path` (image path, relative or absolute)
* `label` (string or int)

If `path` is relative, pass `--base_path /path/to/images_root`. Set accordinly to where you downloaded the images.

## 4) Run training

Recommended:

```bash
./train_uni_nct.sh
```

Or directly:

```bash
python train.py \
  --model_dir foundation_models \
  --splits_dir splits-nct \
  --results_dir training_results \
  --experiment_name uni_nct \
  --num_classes 9 \
  --batch_size 128 \
  --lr 1e-4 \
  --model_freeze
```

Optional:

* Train only some folds: `--folds 0,3`
* Use GPU: `--accelerator gpu` (and optionally `--precision bf16`)
