# AIGC Aesthetics

Code release for a computational aesthetics project on group consensus and individual differences in face and landscape ratings.

The repository is intended to contain analysis code only. Raw ratings, image files, generated feature tables, fitted model traces, figures, reports, and local model checkpoints are intentionally excluded by `.gitignore`.

## Research Questions

1. Which interpretable visual and geometric features explain image-level aesthetic variation?
2. How much additional variance is explained by deep representations such as StyleGAN, InsightFace, DINOv2, and Places365?
3. Do raters show heterogeneous preferences for the same facial or landscape features?

## Main Code Structure

### Face aesthetics pipeline

- `compute_interpretable_features.py`: extract interpretable face geometry and visual features.
- `extract_dinov2.py`: extract DINOv2 embeddings from face images.
- `encoder4editing-main/extract_styleGAN_features.py`, `encoder4editing-main/pca_stylegan_14.py`, `encoder4editing-main/patch_stylegan.py`: project-specific StyleGAN feature helper scripts retained from the local e4e checkout.
- `run_pca_dinov2_14.py`, `run_pca_insightface.py`, `Dinov2_pca_0406.py`, `refit_face_pca_by_variance.py`: reduce deep embeddings to principal components.
- `BYS_kong_model.py`: null Bayesian hierarchical model with rater and image random intercepts.
- `BYS_interpretable_features_model.py`: Bayesian model using 14 interpretable face features.
- `BYS_StyleGAN_features_model.py`, `BYS_Insightface_features_model.py`, `BYS_DINOv2_features_model.py`: deep-feature Bayesian models.
- `BYS_Fusion_28D_Model.py`: fusion model combining interpretable features with deep principal components.
- `BYS_gender_features_model.py`: gender-specific preference interaction model.
- `run_ultimate_heterogeneity.py`: rater-level heterogeneity models for feature preferences.
- `BYS_Face_BlackBox_CV_Ridge_Result/run_face_blackbox_cv_ridge.py`: cross-validated ridge benchmarks for deep representations.

### Rater clustering and heterogeneity visualization

- `cluster_14D_gmm_tsne.py`, `cluster_advanced_hierarchy.py`, `find_optimal_k.py`: cluster rater preference profiles.
- `plot_1300_clustermap.py`, `plot_heterogeneity_quartet.py`, `plot_14D_violin.py`: preference heterogeneity visualizations.
- `draw_enhanced_parallel_coordinates.py`, `draw_face_alluvial_parallel_coordinates.py`, `draw_face_sankey_state_transitions.py`: rater preference state and trajectory plots.
- `BYS_Ultimate_Heterogeneity_Figures/draw_ultimate_heterogeneity_figures.py`: publication-style heterogeneity summary figures.

### Publication figures

- `Picture_code/`: figure scripts for variance decomposition, model comparison, fusion gain, null diagnostics, DINOv2-interpretable bridges, and heterogeneity summaries.
- `Picture_fig3/`: Figure 3 panel-generation scripts for interpretable effects, GAMM curves, and image variance panels.
- `plot_from_GAMM_NonLinear.py`, `generate_gamm_no_rug_no_text_panels.py`: nonlinear effect curve plotting.

### Landscape aesthetics extension

- `Landscape_Features_Model/generate_landscape_ratings_csv.py`: build landscape rating tables from raw survey files.
- `Landscape_Features_Model/BYS_Null_Model/BYS_landscape_null_model.py`: landscape null model.
- `Landscape_Features_Model/BYS_interpretable_Features_Model/`: landscape handcrafted, SegFormer, and depth feature extraction plus interpretable Bayesian models.
- `Landscape_Features_Model/BYS_DINOv2_Features_Model/`: landscape DINOv2 extraction, PCA, and Bayesian model.
- `Landscape_Features_Model/BYS_Places365_Features_Model/`: Places365 feature extraction, PCA, and Bayesian model.
- `Landscape_Features_Model/BYS_StyleGAN_Features_Model/`: StyleGAN feature extraction, PCA, and Bayesian model.
- `Landscape_Features_Model/BYS_Fusion_20D_DINOv2/`, `Landscape_Features_Model/BYS_Fusion_20D_Places365/`, `Landscape_Features_Model/BYS_Fusion_20D_StyleGAN/`: landscape fusion models.
- `Landscape_Features_Model/BYS_Heterogeneity/`: landscape and combined face-landscape heterogeneity analyses.
- `Landscape_Features_Model/Figures_Landscape/`: landscape figure-generation scripts.

## Expected Local Inputs

The public repository does not include data. To run the full pipeline, place local files using the paths referenced by the scripts.

Typical face inputs:

- `ratings_for_bayesian_model.xlsx`
- `face_features.csv`
- `interpretable_face_features.csv`
- `dinov2_features.csv`
- `PCA_14_dinov2.csv`
- `PCA_14_features.csv`
- `PCA_14_stylegan_w.csv`
- local DINOv2 weights in `dinov2-base-local/`
- local InsightFace weights in `insightface-base-local/`

Typical landscape inputs:

- `Landscape_Features_Model/ratings_for_bayesian_model.csv`
- `Landscape_Features_Model/landscape_*_features.csv`
- local model folders under `Landscape_Features_Model/models/`
- optional local `encoder4editing-main/` dependency for StyleGAN-based landscape extraction

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Bayesian models are fitted with Bambi/PyMC and can be computationally expensive. GPU-enabled PyTorch is recommended for deep feature extraction.

## Suggested Workflow

1. Prepare ratings and feature tables.
2. Fit null models with `BYS_kong_model.py` or `Landscape_Features_Model/BYS_Null_Model/BYS_landscape_null_model.py`.
3. Fit interpretable-feature models.
4. Extract deep features, run PCA, and fit deep-feature models.
5. Fit fusion models to compare incremental explanatory power.
6. Fit heterogeneity models and cluster rater preference profiles.
7. Generate publication figures from model summaries and traces.

## GitHub Upload

If this folder is not already a Git repository:

```bash
git init
git remote add origin https://github.com/Chen-backup/AIGC-Aesthetics.git
git branch -M main
git add .
git commit -m "Initial code release"
git push -u origin main
```

If large data files were already tracked in a previous local repository, remove them from the index after updating `.gitignore`:

```bash
git rm -r --cached .
git add .
git commit -m "Keep code only and ignore generated data"
git push -u origin main
```

## Notes

- Generated outputs are reproducible artifacts and should stay out of version control.
- Third-party vendor folders and downloaded model checkpoints are excluded. Install or download them separately when reproducing the full pipeline.
- Some scripts use fixed local paths from the original analysis environment. Adjust these paths before running on a new machine.
