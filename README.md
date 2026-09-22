```markdown
# SpectraFed: Short-Term Traffic Flow Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of SpectraFed, a federated learning framework designed for short-term traffic forecasting. The framework enables collaborative model training across distributed traffic agencies while maintaining data privacy and security.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Reproducing Results](#reproducing-results)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)

## Overview

SpectraFed is a dynamic federated spatiotemporal graph network designed for context-aware traffic forecasting. It addresses critical challenges in intelligent transportation systems by allowing multiple entities to collaboratively train predictive models without sharing raw, sensitive traffic data.

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.8 or higher
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Rasha1990/SpectraFed-short-term-traffic-flow-prediction.git
cd SpectraFed-short-term-traffic-flow-prediction

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
torch>=1.8.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=0.24.0
scipy>=1.7.0
tqdm>=4.62.0
matplotlib>=3.4.0
```

## Quick Start

### Basic Training

```bash
# Train the model on the PEMS08 dataset
python basic_spectrafed.py \
    --dataset PEMS08 \
    --data_dir ./datasets/PEMS08 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --num_clients 5 \
    --communication_rounds 50
```

## Datasets

### Supported Datasets

| Dataset | Sensors | Time Steps | Period | Coverage |
|---------|---------|------------|--------|----------|
| PEMS08 | 170 | 17,856 | 2 months | San Francisco Bay Area |
| PEMS04 | 307 | 16,992 | 2 months | San Francisco Bay Area |

### Dataset Preparation

**PEMS08** (Included in repository):
```bash
# Extract the dataset
unzip PEMS08.zip -d datasets/PEMS08/
```

**PEMS04** (Available via STGODE repository):
- Download from the [STGODE Repository](https://github.com/square-coder/STGODE/tree/main).
- Or request directly from the authors.

### Data Format
- Input: Traffic speed and flow measurements at 5-minute intervals.
- Output: Predicted traffic conditions for subsequent time steps.

## Reproducing Results

### Prior Work Reference

For comparison with our previous work, FedGODE, please visit the [FedGODE Repository](https://github.com/rushaa/FedGODE).

### Reproduction Steps

```bash
# 1. Clone repository
git clone https://github.com/Rasha1990/SpectraFed-short-term-traffic-flow-prediction.git
cd SpectraFed-short-term-traffic-flow-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare datasets
unzip PEMS08.zip -d datasets/PEMS08/

# 4. Run experiments
bash scripts/run_all_experiments.sh
```

## Repository Structure

```text
SpectraFed/
├── basic_spectrafed.py          # Basic implementation
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── models/                      # Model definitions
├── federated/                   # Federated learning client and server logic
├── privacy/                     # Privacy preservation mechanisms
├── datasets/                    # Data loading and preprocessing
├── configs/                     # Configuration files
├── scripts/                     # Utility and experiment scripts
├── results/                     # Experimental results
└── README.md                    # This file
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{alhuthaifi2026spectrafed,
  title={Ultra-Lightweight, Provable, and Secure Dynamic Federated Spatiotemporal Graph Networks for Context-Aware Traffic Forecasting},
  author={Al-Huthaifi, Rasha and Yang, Hailiang and Wang, Hengzhi and Cui, Laizhong and Al-Huda, Zaid},
  journal={Expert Systems with Applications},
  year={2026},
  publisher={Elsevier}
}
```

## Contact

For questions, suggestions, or collaborations, please contact:

- Rasha Al-Huthaifi: rasha.khaled.mosaid@gmail.com
- Hailiang Yang (Corresponding Author): yanghailiang@gml.ac.cn

```
