# Heart Disease ML Pipeline & Streamlit App

A complete, end-to-end machine learning workflow on the UCI Heart Disease dataset, including preprocessing, feature engineering, model training, evaluation, and an interactive Streamlit app for single and batch inference.

## Project Structure

```
Heart_Disease_Project/
├─ data/
│  ├─ heart_disease.csv         # Raw dataset
│  ├─ cleaned_X.csv             # Cleaned, model-ready features (defines input schema)
│  └─ clean_y.csv               # Target labels used during training
├─ models/
│  └─ final_model.pkl           # Trained scikit-learn model artifact
├─ notebooks/                   # EDA, preprocessing, modeling, tuning
│  ├─ 01_data_preprocessing.ipynb
│  ├─ 02_pca_analysis.ipynb
│  ├─ 03_feature_selection.ipynb
│  ├─ 04_supervised_learning.ipynb
│  ├─ 05_unsupervised_learning.ipynb
│  └─ 06_hyperparameter_tuning.ipynb
├─ results/
│  └─ evaluation_metrics.txt    # Saved evaluation metrics
├─ requirements.txt
└─ app.py                       # Streamlit application entry point
```

## Streamlit App Overview

The app loads the trained model from `models/final_model.pkl` and expects input features that match the columns and order of `data/cleaned_X.csv`.

- Single Prediction tab: Fill in patient details; the app builds the one-hot feature vector behind the scenes and predicts risk (and probability if available).
- Batch Prediction tab: Upload a CSV with the exact same header as `data/cleaned_X.csv` to get predictions for many rows at once. A header template download is provided in the UI.
- Documentation tab: Project overview, metrics (from `results/evaluation_metrics.txt` if present), and the exact feature column list.

## Setup

- Recommended Python: 3.11 (Windows supported)
- Create/activate a virtual environment (optional but recommended)

Windows PowerShell example:

```powershell
# From the project root
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the App

From the project root:

```powershell
streamlit run app.py
```

Streamlit will open a local URL in your browser. If it doesn’t, copy the printed URL into your browser.

## Input Schema (very important)

The app derives the exact expected input columns from `data/cleaned_X.csv`. These include:

- Continuous: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`
- One-hot encoded groups (column names start with): `sex_`, `cp_`, `restecg_`, `slope_`, `fbs_`, `exang_`, `thal_`

For batch predictions, the uploaded CSV must have the identical header (names and order). Any extra columns will be ignored; any missing required columns will stop the run.

## Troubleshooting

- UnpicklingError / invalid load key:

  - The model file may have been saved with `joblib`. The loader in `app.py` tries `joblib.load` first, then falls back to `pickle.load`.
  - If it still fails, ensure your environment’s `scikit-learn` version is compatible with the artifact that produced `final_model.pkl`. As a last resort, re-save the model in the current environment:

    ```python
    import joblib
    joblib.dump(model, 'models/final_model.pkl')
    ```

- Model or data file not found:

  - Verify `models/final_model.pkl` and `data/cleaned_X.csv` exist and paths match the project layout.

- Feature mismatch for batch upload:
  - Use the header template download in the Batch tab and populate rows under that header.

## Reproducibility Notes

- Keep `final_model.pkl` and `cleaned_X.csv` in sync with the training pipeline used in the notebooks.
- If you change preprocessing or feature engineering, regenerate both the features and the model.

## License

This project is provided for educational and research purposes. Add a license file if you plan to distribute.
