# Making Euclid VIS Imaging AI-Ready

> A methodology for transforming large-scale astronomical imaging into reusable, task-agnostic representations using Vision Transformers (DINOv2)

---

## 🔭 Overview

Modern astronomical surveys such as **Euclid** are producing imaging data at unprecedented scale. However, transforming raw survey images into scientifically usable representations remains a major bottleneck.

This project presents a **complete AI-ready workflow** that bridges this gap:

* From **Euclid VIS cutouts**
* To **foundation-model embeddings (DINOv2)**
* To **downstream scientific applications**

including:

* Few-label morphology classification
* Anomaly detection at catalog scale
* Similarity-based retrieval

---

## 🧠 Key Idea

Instead of training a model for each task separately, we construct a **reusable representation space**:

> 📌 A single embedding that supports multiple scientific analyses

This aligns with the concept of **AI-ready data products**:

* Task-agnostic
* Scalable
* Interpretable

---

## ⚙️ Workflow

```
Euclid Q1 catalog (Galaxy Zoo)
        ↓
Batch cutout pipeline (NADC)
        ↓
DINOv2 backbone (ViT-S/14)
        ↓
Embedding space (high-dimensional)
        ↓
Applications:
    • Few-label classification
    • LOF anomaly detection
    • Similarity retrieval
```

---

## 🛰️ Dataset

We use the **Galaxy Zoo Euclid (Q1)** dataset:

* ~378,000 galaxies
* Morphology labels derived from Galaxy Zoo votes
* Cross-matched with Euclid VIS imaging

We further construct **high-confidence labels**:

* Remove ambiguous samples
* Retain only reliable morphological categories
* Enable robust supervised evaluation

---

## 🤖 Model

We use the **official DINOv2 backbone**:

* Vision Transformer (ViT-S/14)
* Pretrained on **1+ billion images**
* Strong generalization across domains

Why DINOv2?

* Stable representation (no collapse)
* Strong transfer performance
* No need for domain-specific retraining

---

## 📊 Results

### 🔹 Few-label classification

* Only **1% labeled data**
* Achieves **~95% accuracy**

👉 Demonstrates strong representation quality

---

### 🔹 Anomaly detection

Using LOF in embedding space:

* ~1,600 outliers detected from 370k samples
* Includes:

  * Artifacts
  * Saturated sources
  * Cropping failures
  * Rare morphologies

---

### 🔹 Similarity search

Nearest-neighbor retrieval in embedding space:

* Morphologically consistent results
* Aligns with Galaxy Zoo labels
* Works even without explicit supervision

---

## 📁 Project Structure

```
Making-Euclid-VIS-Imaging-AI-Ready/
├── analyze_embedding/
│   └── gz_analyze_embedding_morphology_regression.py
├── downstream_application/
│   └── gz_analyze_embedding_fullplots_highdim_lof_linear_probe_dinov2.py
│   └── gz_run_fewlabel_benchmark_fixed_loading_official_dinov2_diagnostics_mlp.py
│   └── gz_similarity_search_dinov2.py
├── labels/
│   └── gz_hard_quality_labels.py
├── models/
│   └── download_official_dinov2_backbone.py
│   └── run_training_Euclid_SSL.py
├── outputs/
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

```bash
git clone https://github.com/xiejhhhhhh/Making-Euclid-VIS-Imaging-AI-Ready.git
cd Making-Euclid-VIS-Imaging-AI-Ready

pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Download DINOv2 backbone

```bash
python models/download_official_dinov2_backbone.py
```

---

### 2. Build embedding + UMAP + regression

```bash
python analyze_embedding/gz_analyze_embedding_morphology_regression.py
```

---

### 3. Few-label benchmark

```bash
python labels/gz_hard_quality_labels.py
python models/run_training_Euclid_SSL.py
python downstream_application/gz_run_fewlabel_benchmark_fixed_loading_dinov2_diagnostics_mlp.py
```

---

### 4. Anomaly detection

```bash
python downstream_application/gz_analyze_embedding_fullplots_highdim_lof_linear_probe_dinov2_.py
```

---

### 5. Similarity search

```bash
python downstream_application/gz_similarity_search_dinov2.py
```

---


## 🧪 Scientific Contributions

This project is **not just an application**, but a **methodology**:

### ✔ AI-ready data pipeline

* Standardized image extraction
* Scalable preprocessing
* Reproducible workflow

### ✔ Representation-first paradigm

* Separate representation learning from tasks
* Enable reuse across scientific problems

### ✔ Minimal-label science

* High performance with only 1% labels

### ✔ Survey-scale analysis

* Works on hundreds of thousands of galaxies

---

## ⚠️ Limitations

* Domain-specific SSL model may collapse
* Performance depends on image quality
* Some rare morphologies underrepresented

---

## 📜 Citation

If you use this work, please cite:

```
@article{euclid_ai_ready_2026,
  title={Making Euclid VIS Imaging AI-Ready},
  author={Xie et al.},
  journal={ApJS},
  year={2026}
}
```

---

## 🤝 Acknowledgements

* Euclid Consortium
* Galaxy Zoo
* National Astronomical Data Center (NADC)
* DINOv2 (Meta AI)

---

## 🌌 Final Remark

This work demonstrates that:

> We do not need to build new models for every task.
> We need better representations.

And once the representation is right—

everything else becomes easier.

