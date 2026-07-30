# Personalised Investment Recommendation System using Temporal Graph Networks
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![LightFM](https://img.shields.io/badge/LightFM-Recommender-green)
![TGN](https://img.shields.io/badge/Temporal_Graph_Network-TGN-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> Undergraduate Thesis Project | Computer Science

## Overview

This project is my undergraduate thesis, focusing on developing a personalised investment recommendation system for financial assets. The system recommends suitable investment products to individual users by learning their historical investment behaviour and interactions over time.

Unlike traditional recommendation systems that assume user preferences remain static, this project models dynamic user-item interactions using Temporal Graph Networks (TGN), allowing recommendations to adapt as user interests evolve.

The project evaluates both traditional collaborative filtering methods and graph-based deep learning models to identify the most effective approach for personalised financial recommendations.

---

## Objectives

- Build a personalised investment recommendation system
- Understand user investment behaviour through historical interactions
- Compare traditional and graph-based recommendation models
- Improve recommendation quality using temporal information
- Provide a scalable recommendation framework for financial services

---

## Problem Statement

Investment platforms often recommend products using static recommendation methods that cannot effectively capture changing user preferences over time.

Since investment behaviour continuously evolves due to market conditions and personal financial goals, this project investigates whether temporal graph learning can generate more accurate and personalised recommendations than traditional recommendation algorithms.

---

## Dataset

This project uses the **FAR-Trans** dataset, the first publicly available benchmark dataset for **Financial Asset Recommendation (FAR)** research.

The dataset contains anonymised transaction records collected from a large European financial institution, together with historical asset information and temporal interaction data. Its temporal nature makes it suitable for evaluating both traditional recommendation algorithms and dynamic graph-based models such as Temporal Graph Networks (TGN).

### Dataset Contents

- Investor IDs
- Financial asset IDs
- Transaction timestamps
- Historical user–asset interactions
- Asset information
- Temporal investment behaviour

### Dataset Access

The dataset is publicly available for academic research.

**Official Repository:** https://researchdata.gla.ac.uk/1658/

**DOI:** https://doi.org/10.5525/gla.researchdata.1658

Please download the dataset manually and place it under:

```
data/raw/
```

The dataset is **not included** in this repository due to its large size.
---

## Methodology

The overall workflow is shown below.

```
Raw Data
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Popularity Baseline
    │
    ▼
LightFM Recommendation
    │
    ▼
Temporal Graph Network (TGN)
    │
    ▼
Model Evaluation
    │
    ▼
Recommendation Results
```

---

## Models

### 1. Popularity Model

A simple baseline that recommends the most frequently interacted investment products.

Purpose:

- Baseline comparison
- Fast implementation
- Benchmark performance

---

### 2. LightFM

Hybrid collaborative filtering model combining user-item interactions with metadata.

Advantages:

- Handles sparse datasets
- Learns latent representations
- Strong collaborative filtering baseline

---

### 3. Temporal Graph Network (TGN)

The main model of this thesis.

TGN captures dynamic interactions between users and investment products over time using graph neural networks.

Advantages:

- Models temporal behaviour
- Learns evolving user preferences
- Supports dynamic recommendation scenarios

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- PyTorch Geometric
- LightFM
- Temporal Graph Networks (TGN)
- Jupyter Notebook
- Git
- GitHub

---

## Repository Structure

```
investment-recommendation-system/

│
├── configs/                 # Model configurations
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/               # Experiments
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
│
├── figures/                 # Architecture & diagrams
├── models/                  # Saved models
├── results/                 # Experimental results
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Experimental Pipeline

- Data preprocessing
- Missing value handling
- Feature engineering
- Baseline implementation
- LightFM training
- TGN training
- Hyperparameter tuning
- Model evaluation
- Recommendation generation

---

## Evaluation Metrics

The recommendation models are evaluated using ranking metrics including:

- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- Hit Rate

---

## Results

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---------|-------------|-----------|----------|
| Popularity | - | - | - |
| LightFM | - | - | - |
| TGN | - | - | - |

> Experimental results will be updated after model training.

---

## Future Improvements

Potential future work includes:

- Real-time recommendation
- Portfolio optimisation
- Risk-aware recommendation
- Explainable AI
- Online learning
- REST API deployment
- Web application integration

---

## Installation

Clone the repository

```bash
git clone https://github.com/emmybui/investment-recommendation-system.git

cd investment-recommendation-system
```

Create environment

```bash
pip install -r requirements.txt
```

---

## Run

Example

```bash
python train.py
```

Evaluate

```bash
python evaluate.py
```

---

## Research Contribution

This project investigates the application of temporal graph neural networks in personalised investment recommendation.

The research compares traditional recommendation approaches with temporal graph learning models to better understand how dynamic user behaviour influences recommendation quality in financial applications.

---

## Authors

- **Bui Thi Quynh Nhu**

- **Le Phuong Uyen** 

---
## Contributions

This project was developed collaboratively as an undergraduate thesis.

### Bui Thi Quynh Nhu

- Data preprocessing
- Temporal graph construction
- Temporal Graph Network (TGN)
- Market encoder (TCN)
- Risk-aware fusion
- Multi-task learning
- Evaluation metrics
- Ablation study
- Experimental analysis

### Le Phuong Uyen

- Data loading pipeline
- Baseline models
- Experiment tracking
- Training pipeline optimisation
- Evaluation automation
- API & Dashboard prototype

---
## Supervisor

**PhD. Ho Thi Linh**

---

## License

This repository is intended for academic and portfolio purposes only.

The source code may not be copied, redistributed, or used in other academic submissions without permission from the author.

Copyright © 2026 Bui Thi Quynh Nhu and Le Phuong Uyen.

This repository is part of an undergraduate thesis project.
All rights reserved.

---

## Acknowledgements

I would like to express my sincere gratitude to my supervisor for their continuous guidance and support throughout this research.

Special thanks to Ton Duc Thang University for providing the opportunity and academic environment to conduct this undergraduate thesis.

---
## Citation

If you use the FAR-Trans dataset, please cite the original publication:

### Thesis Repository

```bibtex
@misc{bui2026investment,
  title={Personalised Investment Recommendation System using Temporal Graph Networks},
  author={Bui, Thi Quynh Nhu, Le Phuong Uyen},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/emmybui/investment-recommendation-system}
}
```

### FAR-Trans Dataset

```bibtex
@article{sanzcruzado2024fartrans,
  title={FAR-Trans: An Investment Dataset for Financial Asset Recommendation},
  author={Sanz-Cruzado, Javier and Droukas, Nikolaos and McCreadie, Richard},
  year={2024},
  journal={arXiv preprint arXiv:2407.08692}
}
```
