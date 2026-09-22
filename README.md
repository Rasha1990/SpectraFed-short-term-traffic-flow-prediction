## Prior Work and Datasets

This repository builds upon the authors' foundational work in federated traffic forecasting. For reference, the complete implementation and datasets for our prior work, **FedGODE**, are publicly available here: [FedGODE Repository](https://github.com/rushaa/FedGODE). Consistent with our commitment to reproducibility, this repository provides the foundational architecture for SpectraFed.

To facilitate immediate testing and reproducibility of the basic model, the **PEMS08** dataset is included in this repository, alongside a link to **PEMS04**. PEMS08 is provided directly due to its manageable size, allowing reviewers and researchers to execute the `basic_spectrafed.py` script without extensive data downloading. 

* **PEMS08**: Included in this repository as `PEMS08.zip`. Please extract this file into the `datasets/PEMS08/` directory before running the script. It contains 170 sensors and 17,856 time steps.
* **PEMS04**: Available via the STGODE repository [here](https://github.com/square-coder/STGODE/tree/main) or upon request for full-scale testing.
