# PathoBERT

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

PathoBERT is a deep learning framework for **read-level bacterial pathogenicity prediction** from DNA sequencing reads.

The model combines:

- DNABERT
- LoRA (Low-Rank Adaptation)
- CNN feature extraction
- MCBAM attention
- Multi-Scale Convolutional Attention (MSCA)

to provide robust pathogenicity prediction from short sequencing reads.

---

# Features

- Read FASTA files
- Automatic k-mer tokenization
- DNABERT backbone
- LoRA-based fine-tuning
- CNN + MCBAM + MSCA classifier
- GPU and CPU inference
- Batch inference
- Probability output
- Genome-level prediction summary
- Publication-quality prediction histogram

---

# Installation

## From PyPI

```bash
pip install pathobert
```

## From GitHub

```bash
git clone https://github.com/salimalaarag/PathoBERT.git

cd PathoBERT

pip install .
```

or

```bash
pip install git+https://github.com/salimalaarag/PathoBERT.git
```

---

# Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.6
- transformers
- peft
- numpy
- pandas
- matplotlib
- tqdm

---

# Download DNABERT

DNABERT is required for both tokenization and model inference. The first run requires downloading the complete DNABERT files from Hugging Face, including the tokenizer files and pretrained model weights.

### Automatic download (recommended)

Run the following command once before using PathoBERT:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download("zhihan1996/DNA_bert_6")
print(path)
PY
```
This command downloads the complete DNABERT package from Hugging Face, including:
config.json
pytorch_model.bin
vocab.txt
tokenizer_config.json
special_tokens_map.json
configuration_bert.py
dnabert_layer.py

The downloaded files are stored automatically in the Hugging Face cache:
~/.cache/huggingface/hub/

After this step, PathoBERT can load DNABERT locally.

For offline usage:
For offline inference, enable:

```python
LOCAL_FILES_ONLY=True
```

inside

```
pathobert/config.py
```
Before enabling offline mode, make sure DNABERT has already been downloaded and is available in the local Hugging Face cache.

If LOCAL_FILES_ONLY=True is enabled before downloading DNABERT, PathoBERT will fail during model loading because the tokenizer or pretrained model weights cannot be found.
To verify that DNABERT is available locally:
ls ~/.cache/huggingface/hub/

You should find a directory similar to:
models--zhihan1996--DNA_bert_6

---

# Usage

## Command Line

Example

```bash
python -m pathobert.cli \
    --model Model_checkpoint/final_model.pt \
    --fasta Test_data/Test.fasta \
    --batch_size 64

```

---

# Python API

```python
from pathobert.model import load_model
from pathobert.inference import predict
```

---

# Output

PathoBERT generates

```
output/
│
├── predictions.tsv
├── probs.pt
├── stats.pt
└── Read-level_prediction_distribution.png
```

---

# Model Architecture

```
DNA Reads
      │
      ▼
K-mer Tokenization
      │
      ▼
DNABERT
      │
      ▼
LoRA
      │
      ▼
CNN
      │
      ▼
MCBAM
      │
      ▼
MSCA
      │
      ▼
Classification Head
      │
      ▼
Pathogenicity Probability
```

---

# Repository Structure

```
pathobert/

├── __init__.py
├── analysis.py
├── cli.py
├── config.py
├── dataset.py
├── fasta.py
├── inference.py
├── model.py
└── tokenizer.py
```

---

# Citation

If you use PathoBERT in your research, please cite:

```
Salem A. El-aarag, Mario Flores, Mohamed E Hasan, Alaa E. Hemeida, and Mahmoud ElHefnawi

PathoBERT: 
A Hybrid Attention-Based Genomic Language Model for Read-Level Bacterial Pathogenicity Prediction

(Manuscript in preparation)
```

Update this section after publication.

---

# License

MIT License.

---

# Research Use

PathoBERT is intended **for research purposes only**.

It is **not** intended for:

- clinical diagnosis
- medical decision making
- patient care
- regulatory use

Users are responsible for validating results before applying them in biological or clinical studies.

---

# Author

**Salem A. El-aarag**

Email:
- salem.abdelmonem.stu@gebri.usc.edu.eg
- salim.alaarag@yahoo.com

GitHub

https://github.com/salimalaarag

---

# Acknowledgements

This work uses

- DNABERT
- Hugging Face Transformers
- PEFT
- PyTorch

We thank the developers of these open-source projects.
