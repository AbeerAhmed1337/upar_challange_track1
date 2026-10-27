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
- [`requirements.txt`](./requirements.txt) — Python packages used by the notebook.

## Running the notebook

The notebook is designed to run in **Google Colab with a GPU**; it is not a standalone Python script. It uses Colab-specific shell commands and paths, downloads the UPAR challenge code and datasets, and saves checkpoints and the submission ZIP to Google Drive.

1. Open the [notebook in Google Colab](https://colab.research.google.com/github/AbeerAhmed1337/upar_challange_track1/blob/main/upar_challange_final.ipynb). If that link does not work, download [`upar_challange_final.ipynb`](./upar_challange_final.ipynb) and upload it to Colab.
2. In Colab, select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run the notebook cells **in order**, starting from the top. The notebook installs its Python dependencies and downloads/prepares the datasets.
4. When prompted, authorize Google Drive access. The notebook uses `MyDrive/UPAR2027` to save checkpoints and the final `my_submission.zip`.
5. If the PA100k download fails, follow the notebook's data-download fallback instructions; access to the dataset's Google Drive file may be required.
6. At the end, download `my_submission.zip` from the notebook output or retrieve it from `MyDrive/UPAR2027`.

The notebook installs its dependencies in its setup cells. To install the packages from [`requirements.txt`](./requirements.txt) yourself in Colab, run this in a cell before starting the notebook:

```python
!git clone https://github.com/AbeerAhmed1337/upar_challange_track1.git /content/upar-project
!pip install -r /content/upar-project/requirements.txt
```

PyTorch and torchvision are intentionally not pinned in the requirements file because their compatible versions depend on the runtime and CUDA; Colab provides them preinstalled. The notebook installs `timm==1.0.9` as part of its setup.
