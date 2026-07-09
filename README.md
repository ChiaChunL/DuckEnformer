# 🦆 DuckEnformer

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4.1-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](#installation)
[![Data DOI](https://img.shields.io/badge/data-Zenodo-1682D4.svg?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.21231057)

**DuckEnformer** is a deep learning framework for modeling genome-wide regulatory activity in *duck* (*Anas platyrhynchos*), built on an Enformer-style CNN + Transformer architecture. It performs sequence-to-signal prediction for epigenomic assays such as **ATAC-seq** and **CUT&Tag**, and supports downstream variant-effect (allele-specific imbalance) analysis.

---

## ⚙️ Installation

**System requirements:** Python 3.8 (the pinned `tensorflow==2.4.1` does not provide wheels for Python 3.9+ or for Apple Silicon/arm64 macOS). GPU training/inference additionally requires a matching CUDA/cuDNN install (CUDA 11.0). Tested on: Linux x86_64, Python 3.8, CUDA 11.0.

Typical install time on a normal desktop/workstation with a stable internet connection: **a few minutes** (dominated by downloading the pinned TensorFlow/NumPy/pandas wheels).

**Using [uv](https://github.com/astral-sh/uv) (recommended):**
```bash
git clone https://github.com/ChiaChunL/DuckEnformer.git
cd DuckEnformer
uv sync
uv sync --extra ml   # adds tensorflow / tensorflow-hub
```

**Using pip:**
```bash
git clone https://github.com/ChiaChunL/DuckEnformer.git
cd DuckEnformer
pip install -e ".[ml]"
```

---

## 🧪 Quick start / Demo

A minimal smoke test that builds the DuckEnformer model and runs one forward pass on a randomly generated one-hot DNA sequence. This does **not** require a trained checkpoint, GPU, or any downloaded data/example dataset — it only verifies that the installation is correct and the model builds and runs.

```bash
python scripts/demo_smoke_test.py
```

**Expected output:**
```
Output shape (duck head): (1, 896, 240)
Elapsed: <a few seconds to ~2 minutes on CPU, faster on GPU>
```

For a demo that runs on real (example) data, see [Inference](#inference) below, using the example data under `data/raw/`.

---

## 🗂️ Repository layout

```
DuckEnformer/
├── src/
│   ├── models/          # Enformer model (src/models/enformer.py) & attention modules
│   ├── datasets/        # tf.data pipeline (reads TFRecords produced by data prep)
│   ├── engine/          # Trainer, evaluator, inference helpers
│   └── utils/           # Logging, YAML loading, misc helpers
├── scripts/
│   ├── prepare_data/    # BAM -> BigWig, target/track table construction, TFRecord dataset build
│   ├── model_construct/ # Cluster (SLURM) training launchers
│   ├── inference_model/ # Region / variant (allele-specific) inference
│   └── evaluate_model/  # Held-out Pearson correlation evaluation
├── configs/
│   ├── duck.yaml               # Model & training configuration
│   └── dataset/EpigenomicsData.yaml  # Raw data source paths (edit for your setup)
├── train.py              # Single-GPU training entry point
├── train-multiGPUs.py    # Multi-GPU training entry point (tf.distribute.MirroredStrategy)
├── inference.py          # Standalone inference from a trained checkpoint
├── pyproject.toml
└── LICENSE
```

---

## 📊 Data preparation

DuckEnformer trains on genome-wide signal tracks in BigWig (`.bw`) format, converted to TFRecords.

1. **BAM → BigWig**: `scripts/prepare_data/DuckEpigenomics.bam2bw.py` (or the SLURM wrapper `scripts/prepare_data/runBam2Bw.slurm`).
2. **Track table construction**: `scripts/prepare_data/DuckEpigenomics.createTargets.py` builds the `targets_duck.txt` table consumed by training/inference.
3. **TFRecord dataset construction**: `scripts/prepare_data/split_dataset.sh`.

Step 3 calls [`basenji_data.py`](https://github.com/calico/basenji) from Calico's **basenji** toolkit (Apache-2.0). This tool is **not vendored in this repository**; clone it separately before running the script:

```bash
git clone https://github.com/calico/basenji.git   # pin to a specific commit if reproducing published results
export BASENJI_DIR=/path/to/basenji
```

Then configure the genome FASTA and run:
```bash
FASTA_FILE=/path/to/genome.fa \
BASENJI_DIR="$BASENJI_DIR" \
scripts/prepare_data/split_dataset.sh
```

Update `configs/dataset/EpigenomicsData.yaml` and `configs/duck.yaml` (`data.dataset_root`) to point at your local raw data / TFRecord output before training.

---

## 🚀 Training

Both scripts load their configuration from `<Proj_dir>/configs/duck.yaml` (edit that file directly rather than passing a `--config` flag). `--Proj_dir` defaults to the repository root; override it if running from elsewhere.

**Single GPU:**
```bash
python train.py
```

**Multi-GPU** (`tf.distribute.MirroredStrategy`):
```bash
python train-multiGPUs.py
```

**SLURM cluster:**
```bash
sbatch scripts/model_construct/DuckEnformer-train.slurm
```
(Edit the `PROJECT_ROOT` / `CONDA_ENV` / `CUDA_HOME` variables at the top of the script, or export them beforehand, to match your cluster.)

Both entry points log training/validation loss, evaluation metrics (Pearson correlation), and save checkpoints under `checkpoints/duck_enformer/<run_id>/`.

---

## 🔮 Inference

The trained checkpoint weights (`best/`) used for the results reported in the manuscript are available at Zenodo: [https://doi.org/10.5281/zenodo.21231057](https://doi.org/10.5281/zenodo.21231057). `configs/duck.yaml` in this repository already matches the hyperparameters used to produce this checkpoint. To use it:

```bash
mkdir -p checkpoints/duck_enformer/<run_id>
cp configs/duck.yaml checkpoints/duck_enformer/<run_id>/config.yaml
# download and unzip the Zenodo checkpoint archive into checkpoints/duck_enformer/<run_id>/best/
```

Run prediction from a trained checkpoint:
```bash
python inference.py \
  --ckpt_dir checkpoints/duck_enformer/<run_id>/best \
  --fasta path/to/sequence.fa \
  --out predictions.npz
```

For region- or variant-level inference used in the paper's downstream analysis, see `scripts/inference_model/` (e.g. `region-inference.py`, `deltaAllele-Inference.py`).

---

## 📈 Evaluation

`scripts/evaluate_model/DuckEnformer.evaluate.py` computes held-out (train/valid/test) Pearson correlation and track-wise metrics against the config used for training.

---

## 🔁 Reproducibility

- Config-driven experiments (`configs/duck.yaml`); each run's config is hashed and logged, and a copy is saved next to the checkpoint.
- Dataset construction, training, and evaluation are all parameterized by CLI arguments / YAML config rather than hardcoded paths.
- Data preparation depends on the external `basenji_data.py` tool (see [Data preparation](#data-preparation)); any recent release of [calico/basenji](https://github.com/calico/basenji) is expected to produce equivalent TFRecords for this pipeline.
- Note: training does not currently pin TensorFlow/NumPy random seeds, so re-running from scratch will not reproduce bit-identical weights (only statistically comparable results).

---

## 📦 Data availability

Processed ATAC-seq and CUT&Tag BigWig signal tracks for duck (single archive, `DuckEnformer_bigwig_tracks.tar.gz`), along with the trained model checkpoint used for the results reported in the manuscript, are deposited at Zenodo: [https://doi.org/10.5281/zenodo.21231057](https://doi.org/10.5281/zenodo.21231057). The large result artifacts under `results/` (variant-effect tables, inference outputs) are not distributed in this code repository, but the code to regenerate them from the deposited signal tracks is included here.

## 💻 Code availability

All code required to reproduce the training, inference, and downstream analyses reported in the manuscript is available in this repository under the MIT License. Data preparation additionally depends on Calico basenji toolkit (Apache-2.0, see [Data preparation](#data-preparation)).

---

## 📄 Citation

If you use DuckEnformer in your research, please cite:

```bibtex
@article{duckenformer2026,
  title   = {A multi-tissue three-dimensional atlas of regulatory elements in duck},
  author  = {Yin, Zhongtao and Li, Zhen and Ming, Shao and Wang, Yanan and Li, Jiajun and Lu, Jingsheng and Lu, Xuemei and Wang, Mingshan and Yang, Fangxi and Hao, Jinping and Zhu, Feng and Ye, Shengqiang and Qian, Yunguo and Zhang, Ziding and Yang, Ning and Pan, Zhangyuan and Fang, Lingzhao and Hou, Zhuocheng},
  journal = {Manuscript under review},
  year    = {2026}
}
```

*(This entry will be updated with the final journal, volume, and DOI upon publication.)*

---

## 🙏 Acknowledgements

DuckEnformer's model architecture is adapted from [Enformer](https://github.com/deepmind/deepmind-research/tree/master/enformer) (Avsec et al., *Nature Methods*, 2021), and data preparation builds on tools from the [Basenji](https://github.com/calico/basenji) toolkit (Kelley et al., *Genome Research*, 2018). We thank the original authors for making their code and models publicly available.

---

## 📜 License

This project (code) is released under the MIT License, free for both academic research and commercial use. See [LICENSE](LICENSE) for details.

---

## ✉️ Contact

We welcome inquiries and collaboration opportunities, including extending DuckEnformer to other avian or livestock species, integrating additional epigenomic assays, or other custom applications. Feel free to contact us at ChiaChun.Le@gmail.com.
