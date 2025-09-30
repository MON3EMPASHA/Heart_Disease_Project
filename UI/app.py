import io
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib


MODEL_PATH = os.path.join("models", "final_model.pkl")
CLEANED_X_PATH = os.path.join("data", "cleaned_X.csv")
METRICS_PATH = os.path.join("results", "evaluation_metrics.txt")


@st.cache_data(show_spinner=False)
def load_feature_columns(path: str) -> List[str]:
    sample = pd.read_csv(path, nrows=1)
    return list(sample.columns)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    # Try joblib first (common for scikit-learn models), then fallback to pickle
    try:
        return joblib.load(path)
    except Exception as joblib_err:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as pickle_err:
            raise RuntimeError(
                f"Failed to load model at {path}. Tried joblib and pickle. "
                f"joblib error: {joblib_err}; pickle error: {pickle_err}"
            )


@st.cache_data(show_spinner=False)
def load_text_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def split_feature_groups(feature_columns: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {
        "continuous": [],
        "sex": [],
        "cp": [],
        "restecg": [],
        "slope": [],
        "fbs": [],
        "exang": [],
        "thal": [],
    }
    for col in feature_columns:
        if col in {"age", "trestbps", "chol", "thalach", "oldpeak", "ca"}:
            groups["continuous"].append(col)
        elif col.startswith("sex_"):
            groups["sex"].append(col)
        elif col.startswith("cp_"):
            groups["cp"].append(col)
        elif col.startswith("restecg_"):
            groups["restecg"].append(col)
        elif col.startswith("slope_"):
            groups["slope"].append(col)
        elif col.startswith("fbs_"):
            groups["fbs"].append(col)
        elif col.startswith("exang_"):
            groups["exang"].append(col)
        elif col.startswith("thal_"):
            groups["thal"].append(col)
    # Preserve the original order within each group
    for key in groups:
        groups[key] = [c for c in feature_columns if c in groups[key]]
    return groups


def build_input_vector(
    feature_columns: List[str],
    groups: Dict[str, List[str]],
    user_inputs: Dict[str, float | int | str | bool],
) -> pd.DataFrame:
    values: Dict[str, float] = {c: 0.0 for c in feature_columns}

    # Continuous
    for c in groups["continuous"]:
        values[c] = float(user_inputs[c])

    # One-hot groups
    def set_one_hot(group_name: str, selected_suffix: str | int | float | bool):
        cols = groups[group_name]
        if not cols:
            return
        # Normalize selection for matching (strings like "3.0" vs 3)
        selected = str(selected_suffix)
        for col in cols:
            suffix = col.split("_", 1)[1]
            values[col] = 1.0 if str(suffix) == selected else 0.0

    set_one_hot("sex", user_inputs.get("sex"))
    set_one_hot("cp", user_inputs.get("cp"))
    set_one_hot("restecg", user_inputs.get("restecg"))
    set_one_hot("slope", user_inputs.get("slope"))
    set_one_hot("fbs", user_inputs.get("fbs"))
    set_one_hot("exang", user_inputs.get("exang"))
    set_one_hot("thal", user_inputs.get("thal"))

    return pd.DataFrame([values], columns=feature_columns)


def render_single_prediction_ui(feature_columns: List[str]) -> Tuple[pd.DataFrame | None, bool]:
    groups = split_feature_groups(feature_columns)

    st.subheader("Patient Information")

    left, right = st.columns(2)

    with left:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", 70, 250, 130)
        chol = st.number_input("Cholesterol (chol)", 100, 700, 240)
    with right:
        thalach = st.number_input("Max Heart Rate (thalach)", 60, 250, 150)
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        ca = st.number_input("# of Major Vessels (ca)", 0, 4, 0)

    st.subheader("Categorical Features")

    # Helpers to extract options from available one-hot columns
    def options_from_group(cols: List[str]) -> List[str]:
        return [c.split("_", 1)[1] for c in cols]

    sex_opt = st.selectbox("Sex", options_from_group(groups["sex"]))
    cp_opt = st.selectbox("Chest Pain Type (cp)", options_from_group(groups["cp"]))
    restecg_opt = st.selectbox("Resting ECG (restecg)", options_from_group(groups["restecg"]))
    slope_opt = st.selectbox("Slope of ST Segment (slope)", options_from_group(groups["slope"]))
    fbs_opt = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options_from_group(groups["fbs"]))
    exang_opt = st.selectbox("Exercise-induced Angina (exang)", options_from_group(groups["exang"]))
    thal_opt = st.selectbox("Thalassemia (thal)", options_from_group(groups["thal"]))

    if st.button("Predict", type="primary"):
        user_inputs = {
            "age": age,
            "trestbps": trestbps,
            "chol": chol,
            "thalach": thalach,
            "oldpeak": oldpeak,
            "ca": ca,
            "sex": sex_opt,
            "cp": cp_opt,
            "restecg": restecg_opt,
            "slope": slope_opt,
            "fbs": fbs_opt,
            "exang": exang_opt,
            "thal": thal_opt,
        }
        features_df = build_input_vector(feature_columns, groups, user_inputs)
        return features_df, True

    return None, False


def render_batch_prediction_ui(feature_columns: List[str]) -> pd.DataFrame | None:
    st.info("Upload a CSV with the exact same columns as `cleaned_X.csv` (one-hot schema).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.download_button(
            label="Download feature columns template",
            data=",".join(feature_columns).encode("utf-8"),
            file_name="feature_columns_template.csv",
            mime="text/csv",
        )
        return None

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

    missing = [c for c in feature_columns if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_columns]
    if missing:
        st.error(f"CSV missing required columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        return None
    if extra:
        st.warning(f"CSV has unexpected columns which will be ignored: {extra[:10]}{'...' if len(extra) > 10 else ''}")
        df = df[feature_columns]

    return df


def main():
    st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")
    st.title("Heart Disease Risk Predictor")
    st.caption("Using the trained model from the UCI Heart Disease dataset")

    # Validate resources
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()
    if not os.path.exists(CLEANED_X_PATH):
        st.error(f"Feature file not found at {CLEANED_X_PATH}")
        st.stop()

    # Load model and feature schema
    feature_columns = load_feature_columns(CLEANED_X_PATH)
    model = load_model(MODEL_PATH)

    tab_single, tab_batch, tab_docs = st.tabs(["Single Prediction", "Batch Prediction", "Documentation"])

    with tab_single:
        single_df, ready = render_single_prediction_ui(feature_columns)
        if ready and single_df is not None:
            try:
                pred = model.predict(single_df)[0]
                # Try probas if available
                proba_text = ""
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(single_df)[0]
                    positive_index = 1 if len(proba) > 1 else 0
                    risk = float(proba[positive_index])
                    proba_text = f" (risk probability: {risk:.3f})"
                st.success(f"Prediction: {'Disease' if int(pred)==1 else 'No Disease'}{proba_text}")
                with st.expander("Model Inputs"):
                    st.dataframe(single_df.T, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab_batch:
        df = render_batch_prediction_ui(feature_columns)
        if df is not None and st.button("Run Batch Predictions"):
            try:
                preds = model.predict(df)
                out = df.copy()
                out["prediction"] = preds
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(df)
                    positive_index = 1 if proba.shape[1] > 1 else 0
                    out["risk_probability"] = proba[:, positive_index]
                st.dataframe(out.head(50), use_container_width=True)
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

    with tab_docs:
        st.header("Project Documentation")
        st.markdown(
            """
            ### Overview
            This Streamlit app predicts heart disease risk using a trained scikit-learn model built on the UCI Heart Disease dataset. It supports:
            - Single prediction via a guided form that maps inputs to the one-hot-encoded feature schema
            - Batch prediction via CSV upload aligned to the feature columns in `data/cleaned_X.csv`

            ### Data
            - Source: UCI Heart Disease dataset
            - Cleaned features are stored in `data/cleaned_X.csv` and define the exact input schema expected by the model.

            ### Preprocessing & Feature Schema
            - Continuous features: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`
            - Categorical features are one-hot encoded (e.g., `sex_0`, `sex_1`, `cp_*`, `restecg_*`, `slope_*`, `fbs_*`, `exang_*`, `thal_*`).
            - The app reads the first row of `cleaned_X.csv` to determine the exact feature columns and their order.

            ### Modeling
            - The final trained model is loaded from `models/final_model.pkl`.
            - The loader tries `joblib.load` first, then falls back to `pickle.load` for compatibility.

            ### How to Use This App
            1. Go to the Single Prediction tab and fill in patient information and categorical selections. Click Predict to view the result and probability if available.
            2. For batch inference, prepare a CSV with the exact same columns as in `data/cleaned_X.csv`. Use the template download in the Batch tab. Upload the CSV and click Run Batch Predictions.

            ### Model Performance
            The metrics below are loaded from `results/evaluation_metrics.txt` if available.
            """
        )

        metrics_text = load_text_file(METRICS_PATH)
        if metrics_text:
            st.code(metrics_text)
        else:
            st.info("Evaluation metrics file not found. Train the model and export metrics to `results/evaluation_metrics.txt`.")

        with st.expander("Feature Columns (from cleaned_X.csv)"):
            st.write("The app expects the following columns in this exact order:")
            st.code("\n".join(feature_columns))

        st.markdown(
            """
            ### Limitations & Notes
            - Predictions are based on historical data and a simplified feature set; they do not constitute medical advice.
            - Ensure your runtime scikit-learn version is compatible with how the model was saved. If loading fails, re-save the model in the current environment using `joblib.dump`.
            - For reproducibility, keep `cleaned_X.csv` and `final_model.pkl` in sync with the training pipeline.
            """
        )


if __name__ == "__main__":
    main()


