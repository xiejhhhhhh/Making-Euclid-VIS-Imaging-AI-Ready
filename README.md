# Making Euclid VIS Imaging AI-Ready  
### A Self-Supervised Foundation Model for Morphology, Data Quality, and Similarity Search

This repository provides the official implementation accompanying the paper:

**Making Euclid VIS Imaging AI-Ready: A Self-Supervised Foundation Model for Morphology, Data Quality, and Similarity Search**  
*(submitted to ApJS)*

The goal of this project is to transform Euclid VIS imaging data into an **AI-ready representation** using self-supervised learning, enabling scalable, label-free analysis of morphology, data quality, and source similarity at survey scale.

---

## 🔭 Motivation

Modern astronomical surveys such as **Euclid** produce imaging data at volumes that exceed the capacity of traditional, catalog-driven analysis pipelines. While Euclid VIS delivers stable, high-resolution, space-based imaging, raw images and handcrafted catalog features are **not inherently AI-ready**.

This project addresses that gap by:

- Learning directly from **pixel-level Euclid VIS cutouts**
- Avoiding **labels, catalog features, and task-specific supervision**
- Producing a **reusable representation layer** that supports multiple downstream applications without retraining

The learned embedding is designed to serve as a **foundation representation**, not a classifier.

---

## ✨ Key Contributions

- **Self-supervised Euclid VIS foundation model**  
  A DINO-based Vision Transformer trained on unlabeled VIS cutouts.

- **Continuous morphology manifold**  
  The learned embedding organizes sources along a smooth, non-linear manifold reflecting intrinsic morphological variation.

- **AI-ready applications without retraining**  
  - Data-driven anomaly detection  
  - Morphology-based similarity search

These applications are demonstrations of utility rather than task-optimized endpoints.

---

## 🧠 What Does “AI-ready” Mean Here?

In this work, *AI-ready* refers to:

- **Standardized inputs** (fixed-size, normalized VIS cutouts)
- **Label-free representation learning**
- **Stable embedding space** usable across tasks
- **Decoupling representation from downstream objectives**

> **AI-ready representation ≠ downstream classifier**

The embedding is intended as a common interface between Euclid VIS data and future AI-driven science.

---

## 📦 Repository Structure

```text
.
├── data/
│   └── example_cutouts/        # Example Euclid VIS cutouts (64×64)
│
├── preprocessing/
│   ├── Euclid_flash_app.py     # Cutout construction & normalization
│   └── Euclid_100k_startable.py# Quality-aware data cleaning
│
├── training/
│   └── run_training.py         # DINO ViT-2 self-supervised training
│
├── analysis/
│   ├── analyze_embedding.py    # UMAP / PCA projections & proxy analysis
│   └── validate_model_outliers.py # Anomaly detection & validation
│
├── retrieval/
│   └── similarity_search.py    # Embedding-based nearest-neighbour search
│
├── figures/                    # Figures used in the paper
│
└── README.md
