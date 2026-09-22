## Prior Work and Datasets

This repository builds upon the authors' foundational work in federated traffic forecasting. For reference, the complete implementation and datasets for our prior work, **FedGODE**, are publicly available here: [https://github.com/rushaa/FedGODE]. Consistent with that commitment to reproducibility, this repository provides the foundational architecture for SpectraFed.

To facilitate immediate testing and reproducibility of the basic model, the **PEMS08** dataset (and links to **PEMS04**) are provided in the `datasets/` directory. PEMS08 is included directly due to its manageable size, allowing reviewers and researchers to execute the `basic_spectrafed.py` script without extensive data downloading.

* **PEMS08**: Included in this repository (`datasets/pems08/`). Contains 170 sensors and 17,856 time steps.
* **PEMS04**: Available via the STGODE repository [https://github.com/square-coder/STGODE/tree/main] or upon request for full-scale testing.
