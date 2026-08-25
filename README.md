# AIGC Aesthetics

This repository contains the core analysis code for an AIGC aesthetics project on face and landscape ratings.

The public version keeps only data processing, feature extraction, dimensionality reduction, Bayesian modeling, fusion modeling, and rater-heterogeneity computation code. Raw data, generated feature tables, fitted traces, figures, result folders, downloaded model weights, and plotting scripts are intentionally excluded.

## Scope

- Face aesthetics: interpretable facial features, DINOv2, InsightFace, StyleGAN, null models, interpretable models, deep models, fusion models, and rater preference heterogeneity.
- Landscape aesthetics: handcrafted, semantic, depth, DINOv2, Places365, StyleGAN, null/interpretable/deep/fusion models, and heterogeneity modeling.
- Benchmarks: PCA utilities and cross-validated ridge-style model comparison code where applicable.

## Main Files

- `compute_interpretable_features.py`: face interpretable feature extraction.
- `extract_dinov2.py`: face DINOv2 embedding extraction.
- `run_pca_dinov2_14.py`, `run_pca_insightface.py`, `refit_face_pca_by_variance.py`: face PCA utilities.
- `BYS_kong_model.py`: face null Bayesian hierarchical model.
- `BYS_interpretable_features_model.py`: face interpretable-feature model.
- `BYS_StyleGAN_features_model.py`, `BYS_Insightface_features_model.py`, `BYS_DINOv2_features_model.py`: face deep-feature models.
- `BYS_Fusion_28D_Model.py`: face interpretable + deep fusion model.
- `run_ultimate_heterogeneity.py`: face rater-heterogeneity modeling.
- `Landscape_Features_Model/`: landscape feature extraction and modeling code.
- `encoder4editing-main/`: only project-specific StyleGAN helper scripts are retained.

## Data Policy

No research data or generated results are included in this repository. To reproduce the analyses, prepare the required local inputs referenced by each script, such as rating tables, image-level feature tables, PCA feature tables, and local model checkpoints.

Common local inputs include:

- `ratings_for_bayesian_model.xlsx`
- `face_features.csv`
- `interpretable_face_features.csv`
- `PCA_14_dinov2.csv`
- `PCA_14_features.csv`
- `PCA_14_stylegan_w.csv`
- `Landscape_Features_Model/ratings_for_bayesian_model.csv`
- local model folders such as `dinov2-base-local/`, `insightface-base-local/`, and landscape model checkpoints

## Installation

```bash
python -m venv .venv
pip install -r requirements.txt
```

Bayesian models use Bambi/PyMC and may require substantial CPU time and memory. Deep feature extraction benefits from a GPU-enabled PyTorch installation.

## Notes

- Some scripts still contain local path assumptions from the original analysis environment.
- Plotting scripts and generated result folders are kept out of the public GitHub version.
- The repository is intended as a code archive for methods and reproducibility, not as a complete runnable package.
