# Copilot Instructions for VELIB_PROJ

## Project Overview
- This project analyzes and predicts Paris Vélib' bike counts using data science and machine learning workflows.
- The architecture is modular: data ingestion/preprocessing (`data/`), model training/evaluation/prediction (`models/`), Streamlit app (`app.py`), and utility scripts (`utils/`, `scripts/`).
- Data flows from ingestion (scripts) → preprocessing (data/) → modeling (models/) → visualization/UI (app.py, pages/).

## Key Workflows
- **Update data:** Run `python scripts/update_data.py` to fetch and append new raw data. This script manages metadata and ensures up-to-date datasets. Also available: `make update` or `scripts/run_update.sh`.
- **Train model:** Run `python models/train.py --data <raw_csv>` to train and save a model (`artifacts/model.pkl`). Also available: `make train` or `scripts/run_train.sh <data.csv> <artifacts/model.pkl>`.
- **Predict:** Run `python models/predict.py --data <raw_csv> --model <model.pkl>` to generate predictions (`artifacts/predictions.csv`). Also available: `make predict` or `scripts/run_predict.sh`.
- **Evaluate:** Run `python models/evaluation.py --data <raw_csv> --model <model.pkl>` to compute metrics (`artifacts/eval_metrics.json`). Also available: `make eval` or `scripts/run_eval.sh`.
- **Streamlit app:** Launch with `streamlit run app.py` for interactive analysis and visualization.

## Project Conventions & Patterns
- **Data:** Main raw data file is `comptage_velo_donnees_compteurs.csv` (created/updated by scripts). Data loading and preprocessing are cached for performance (see `@st.cache_data`).
- **Preprocessing:** All feature engineering and cleaning logic is in `data/preprocessing.py`. Use `preprocess_data()` for consistency.
- **Model I/O:** Use `my_utils.save_model` and `my_utils.load_model` for model persistence (typically as `artifacts/model.pkl`).
- **Metrics:** Evaluation metrics are output as JSON for easy downstream use.
- **Paths:** Output artifacts (models, metrics, predictions) are saved in an `artifacts/` directory (created as needed).
- **Scripts:** All CLI scripts use `argparse` for arguments and are runnable as standalone modules.
- **Visualization:** All plots/images are saved to `assets/plots/` and reused if present (see `ensure_png_*` functions).

## Integration & Dependencies
- **External libraries:** Key dependencies include `streamlit`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `folium`, `statsmodels`, `scipy`, and `lightgbm`.
- **System dependencies:** LightGBM requires the OpenMP runtime on some platforms. Common install commands:
  - macOS (Homebrew): `brew install libomp`
  - Debian/Ubuntu: `sudo apt-get install libomp-dev`
  - Conda: `conda install -c conda-forge libomp`
  After installing the system lib, reinstall `lightgbm` in your virtualenv if needed: `pip install --no-binary :all: lightgbm` or `pip install lightgbm`.
- **No requirements.txt:** Install dependencies manually or infer from imports in codebase.
- **No test suite:** No automated tests are present; validate changes by running scripts and the app.

## Examples
- Update data: `python scripts/update_data.py`
- Train: `python models/train.py --data comptage_velo_donnees_compteurs.csv`
- Predict: `python models/predict.py --data comptage_velo_donnees_compteurs.csv --model artifacts/model.pkl`
- Evaluate: `python models/evaluation.py --data comptage_velo_donnees_compteurs.csv --model artifacts/model.pkl`
- Run app: `streamlit run app.py`

## Key Files & Directories
- `data/` — ingestion, preprocessing, and metadata logic
- `models/` — training, prediction, and evaluation scripts
- `scripts/update_data.py` — fetches and appends new data
- `app.py` — Streamlit UI entry point
- `assets/plots/` — generated plots for reuse in app
- `pages/` — additional Streamlit app pages

---
_If any conventions or workflows are unclear, please review the relevant script or ask for clarification._
