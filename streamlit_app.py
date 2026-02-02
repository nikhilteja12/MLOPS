import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from my_utils import preprocess_data, load_model


st.set_page_config(page_title="Traffic Count Predictor", layout="wide")
st.title("🚲 Comptage Horaire - Prediction App")

model_path = st.text_input("Model path", value="artifacts/model.pkl")

uploaded = st.file_uploader("Upload raw CSV", type=["csv"])

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)

    st.subheader("Raw data preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    with st.spinner("Preprocessing... (weather API call included)"):
        df_encoded, features = preprocess_data(df_raw)

    st.success(f"Preprocessing done  Rows after feature engineering: {len(df_encoded)}")

    with st.spinner("Loading model and predicting..."):
        model = load_model(model_path)
        X = df_encoded[features]
        preds = model.predict(X)

    result = df_encoded[["date_et_heure_de_comptage", "identifiant_du_site_de_comptage"]].copy()
    result["prediction_comptage_horaire"] = preds

    st.subheader("Predictions preview")
    st.dataframe(result.head(50), use_container_width=True)

    if "comptage_horaire" in df_encoded.columns:
        y = df_encoded["comptage_horaire"].values
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)

        st.subheader("Evaluation (if target available)")
        st.json({"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)})

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV to start.")
