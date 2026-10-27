# UPAR Challenge 2027 — Pedestrian Attribute Recognition

This repository contains my final notebook submission for the **UPAR Challenge 2027** pedestrian attribute recognition task.

## Result

### Official leaderboard — test set

| Avg | mA | Label_F1 | Inst_Acc | Inst_Prec | Inst_Rec | Inst_F1 |
|---:|---:|---:|---:|---:|---:|---:|
| **0.7696** | 0.7737 | 0.5569 | 0.6482 | 0.7420 | 0.8042 | 0.7655 |

Submitted by **abeer_ahmed** on **2026-10-26** (submission ID `1010213`, group `1010213_VID-Docker-GPU`).

### Notebook holdout results

| Evaluation | Score |
|---|---:|
| Challenge_Avg with per-attribute tuned thresholds | **90.60%** |
| Challenge_Avg using a fixed 0.5 threshold | **90.05%** |

The holdout scores above are local validation results from the notebook; they are separate from the official test-set leaderboard results.

## Approach

- Fine-tuned an ImageNet-pretrained **ConvNeXt-Base** (`convnext_base.fb_in22k_ft_in1k`) to predict the challenge's 40 binary pedestrian attributes.
- Trained on the UPAR training data from **Market1501, PA100k, and PETA**, using the remaining validation images for training. Held out 2,000 validation images per dataset (6,000 total) for model selection and threshold tuning.
- Trained for three epochs at 256 × 128 resolution with AdamW, masked binary cross-entropy, class-imbalance weighting, label smoothing, and an exponential moving average of model weights.
- Used horizontal flips, color jitter, and low-resolution/JPEG augmentations to improve robustness to surveillance-image quality.
- Selected a separate decision threshold for each attribute by optimizing the mean per-dataset `Challenge_Avg` on the held-out validation set. The challenge score is computed from mean accuracy and instance F1.
- Packaged an inference module with batched prediction and horizontal-flip test-time augmentation.

The notebook also tests the packaged inference code in a clean environment: it reports an output shape of `(8, 40)`, a maximum probability difference of `0.000736` versus the training environment, and complete agreement after binarization on that eight-image check.

## Contents

- [`upar_challange_final.ipynb`](./upar_challange_final.ipynb) — the final Colab notebook, including data preparation, training, holdout evaluation, packaging, and inference validation.

## Running the notebook

The notebook is designed for Google Colab with a GPU. It downloads the UPAR challenge resources, reads the dataset files, and uses Google Drive for checkpoints and submission artifacts. Dataset access and availability may depend on the upstream download links and the user's Drive setup.
