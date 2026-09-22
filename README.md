# SpectraFed: Ultra-Lightweight, Provable, and Secure Dynamic Federated Spatiotemporal Graph Networks for Context-Aware Traffic Forecasting

This repository contains the official implementation of SpectraFed, a dynamic federated spatiotemporal graph network designed for context-aware short-term traffic forecasting. The framework enables collaborative model training across distributed traffic agencies while maintaining data privacy and security.

## Note on Code Availability
A portion of the basic source code for SpectraFed has been uploaded to this repository to provide a foundational overview of the framework architecture. The complete source code, including all training scripts, advanced privacy modules, and detailed preprocessing pipelines, will be made publicly available upon the official acceptance of our manuscript. For the duration of the peer-review process, the full codebase has been provided to the journal editors and reviewers as part of the submission package to ensure transparency and reproducibility.

## Dataset Information
SpectraFed was evaluated on publicly available traffic forecasting datasets. We do not host the raw data in this repository due to size constraints, but the official sources are listed below. Preprocessing and stratified splitting scripts will be included in the final release.

* **PEMS08**: Available via standard traffic forecasting dataset repositories (a basic version is included in this repository for initial testing).
* **PEMS04**: Official STGODE Repository (https://github.com/square-coder/STGODE/tree/main) or by direct request from the authors.

## Reproduction and Setup
Upon public release, this repository will include:

* A `requirements.txt` file for dependency management.
* Step-by-step scripts for data preprocessing, non-IID Dirichlet client partitioning, and federated model training.
* Configuration files matching the hyperparameters reported in the manuscript.

## Citation
If you find this work or the provided code useful for your research, please cite our paper:

```bibtex
@article{alhuthaifi2026spectrafed,
  title={Ultra-Lightweight, Provable, and Secure Dynamic Federated Spatiotemporal Graph Networks for Context-Aware Traffic Forecasting},
  author={Al-Huthaifi, Rasha and Yang, Hailiang and Wang, Hengzhi and Cui, Laizhong and Al-Huda, Zaid},
  journal={Expert Systems with Applications},
  year={2026},
  note={Submitted for publication},
  publisher={Elsevier}
}
