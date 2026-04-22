# Making Euclid VIS Imaging AI-Ready

This project packages the core code and paper materials for the study **Making Euclid VIS Imaging AI-Ready**.
It focuses on turning Euclid VIS galaxy cutouts into a reusable embedding space using the official **DINOv2 ViT-S/14** backbone, then using that representation for:

- morphology manifold analysis
- physical-parameter regression probes
- few-label morphology classification
- LOF-based anomaly detection
- similarity-based retrieval

## Project scope

The project implements an **AI-ready workflow** rather than a new backbone model:
1. standardized Euclid VIS cutout generation and ingestion
2. official DINOv2 checkpoint download and reuse
3. frozen-embedding construction at survey scale
4. downstream probes for morphology and physical structure
5. anomaly detection and retrieval in embedding space

## Directory layout

```text
.
├── scripts/
│   ├── download_official_dinov2_backbone.py
│   ├── gz_HQ_labels.py
│   ├── gz_analyze_embedding_morphology_regression_official_v1.py
│   ├── analyze_embedding_fullplots_highdim_lof_linear_probe_official_dinov2.py
│   ├── gz_run_fewlabel_benchmark_fixed_loading_official_dinov2_v2_diagnostics_mlp.py
│   └── gz_similarity_search_improved_official_dinov2_fixed.py
├── docs/
│   ├── Euclid_DinoV2VIT_ApJS.docx
│   └── README_cutout_service_CN.md
├── figures/
│   ├── Figure6_fewlabel.png
│   ├── Figure7_similarity_1.png
│   ├── Figure8_similarity_2.png
│   ├── Figure9_seed_dynamics.png
│   └── Figure10_linear_vs_mlp.png
├── configs/
│   └── paths.example.json
├── requirements.txt
├── .gitignore
└── CITATION.cff
```

## Core scripts

### 1) `download_official_dinov2_backbone.py`
Downloads the official DINOv2 model from `torch.hub` and saves a clean backbone checkpoint for reproducible use.

### 2) `gz_HQ_labels.py`
Converts probabilistic Galaxy Zoo morphology outputs into high-confidence hard labels for downstream classification.

### 3) `gz_analyze_embedding_morphology_regression_official_v1.py`
Builds DINOv2 embeddings, runs UMAP, regression probes, morphology probes, clustering, retrieval examples, and anomaly example generation.

### 4) `analyze_embedding_fullplots_highdim_lof_linear_probe_official_dinov2.py`
Produces full embedding diagnostics including 2D/3D UMAP, LOF outliers, physical-correlation plots, and outlier FITS galleries.

### 5) `gz_run_fewlabel_benchmark_fixed_loading_official_dinov2_v2_diagnostics_mlp.py`
Runs the few-label benchmark with three initialization strategies (`random`, `imagenet_dinov2`, `euclid_ssl`) and two probe heads (`linear`, `mlp`), while also logging embedding-health diagnostics.

### 6) `gz_similarity_search_improved_official_dinov2_fixed.py`
Performs morphology-aware similarity retrieval in the official DINOv2 embedding space using cosine similarity plus label/physical priors.

## Environment

Python 3.10+ is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data and path configuration

Most scripts currently contain **hard-coded local Windows paths** from the research environment.
Before running on a new machine, update:
- `DATA_ROOT`
- `CATALOG_PATH`
- `MODEL_PATH`
- `OUTPUT_DIR`

A template file is provided in `configs/paths.example.json`.

## Suggested execution order

1. Prepare or verify Euclid VIS cutouts
2. Download the official DINOv2 checkpoint
3. Generate high-confidence labels
4. Build embeddings and morphology/physics analyses
5. Run few-label classification benchmarks
6. Run similarity retrieval and anomaly analysis

## Example workflow

```bash
python scripts/download_official_dinov2_backbone.py
python scripts/gz_HQ_labels.py
python scripts/gz_analyze_embedding_morphology_regression_official_v1.py
python scripts/gz_run_fewlabel_benchmark_fixed_loading_official_dinov2_v2_diagnostics_mlp.py
python scripts/gz_similarity_search_improved_official_dinov2_fixed.py
```

## Notes

- Official DINOv2 expects RGB input; the pipeline replicates grayscale Euclid VIS cutouts across 3 channels.
- The project is built around **representation reuse**, not Euclid-specific backbone retraining.
- The Euclid SSL comparison is kept for diagnostic and discussion purposes.
- Some scripts depend on the external `Euclid_DINOv2_VIT` codebase (`euclid_dino.*` imports). Keep that package available in the Python path when running locally.

## Citation

If you use this project, please cite the associated paper and the upstream resources:
- Euclid Q1 data products
- Galaxy Zoo Euclid (Q1)
- DINOv2
- NADC Euclid cutout service

