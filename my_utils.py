import pandas as pd
import requests
import numpy as np


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from category_encoders.target_encoder import TargetEncoder
from lightgbm import LGBMRegressor
import pickle
from sklearn.pipeline import Pipeline
# import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ===========================================================
#  LOAD & SAVE MODEL
# ===========================================================
def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


# ===========================================================
#  WEATHER API (Open-Meteo archive)
# ===========================================================
def query_weather_api(df, latitude, longitude, start_date, end_date):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
        "&timezone=Europe%2FParis"
    )

    response = requests.get(url)

    if response.status_code != 200:
        print("⚠️ Weather API error:", response.status_code)
        return None

    data = response.json()
    if "hourly" not in data:
        return None

    weather_df = pd.DataFrame(data["hourly"])
    weather_df.rename(columns={"time": "date_et_heure_de_comptage"}, inplace=True)
    weather_df["date_et_heure_de_comptage"] = pd.to_datetime(weather_df["date_et_heure_de_comptage"])

    return weather_df


# ===========================================================
#  PREPROCESS FUNCTION
# ===========================================================
def preprocess_data(df):
    df = df.copy()

    df["date_et_heure_de_comptage"] = pd.to_datetime(df["date_et_heure_de_comptage"], errors="coerce")
    df = df.dropna(subset=["date_et_heure_de_comptage"])

    df["hour"] = df["date_et_heure_de_comptage"].dt.hour
    df["day"] = df["date_et_heure_de_comptage"].dt.day
    df["month"] = df["date_et_heure_de_comptage"].dt.month
    df["weekday"] = df["date_et_heure_de_comptage"].dt.weekday
    df["year"] = df["date_et_heure_de_comptage"].dt.year

    # season
    df["season"] = df["month"] % 12 // 3 + 1

    # rush hours
    df["is_rush_hour"] = df["hour"].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)

    # night
    df["is_night"] = df["hour"].apply(lambda x: 1 if (x <= 5) or (x >= 22) else 0)

    # weekend
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)

    # holiday (simple)
    df["is_holiday"] = df["month"].apply(lambda x: 1 if x == 8 else 0)

    # extract lat/lon from coordinates
    if "coordonnées_géographiques" in df.columns:
        coords = df["coordonnées_géographiques"].astype(str).str.split(",", expand=True)
        df["latitude"] = pd.to_numeric(coords[0], errors="coerce")
        df["longitude"] = pd.to_numeric(coords[1], errors="coerce")

    # weather merge
    if "latitude" in df.columns and "longitude" in df.columns:
        lat = df["latitude"].dropna().iloc[0] if df["latitude"].notna().any() else None
        lon = df["longitude"].dropna().iloc[0] if df["longitude"].notna().any() else None

        if lat is not None and lon is not None:
            start_date = df["date_et_heure_de_comptage"].dt.date.min().strftime("%Y-%m-%d")
            end_date = df["date_et_heure_de_comptage"].dt.date.max().strftime("%Y-%m-%d")

            weather_df = query_weather_api(df, lat, lon, start_date, end_date)

            if weather_df is not None:
                df = df.merge(weather_df, on="date_et_heure_de_comptage", how="left")

    # missing weather fill
    for c in ["temperature_2m", "precipitation", "wind_speed_10m"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # sort
    df = df.sort_values(["identifiant_du_site_de_comptage", "date_et_heure_de_comptage"])

    # lags and rolling
    df["lag_1"] = df.groupby("identifiant_du_site_de_comptage")["comptage_horaire"].shift(1)
    df["lag_24"] = df.groupby("identifiant_du_site_de_comptage")["comptage_horaire"].shift(24)
    df["rolling_mean_24"] = (
        df.groupby("identifiant_du_site_de_comptage")["comptage_horaire"]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    df["lag_1"] = df["lag_1"].fillna(df["lag_1"].median())
    df["lag_24"] = df["lag_24"].fillna(df["lag_24"].median())
    df["rolling_mean_24"] = df["rolling_mean_24"].fillna(df["rolling_mean_24"].median())

    # cyclic encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # drop unused columns
    cols_to_drop = [
        "identifiant_technique_compteur",
        "mois_annee_comptage",
        "identifiant_du_compteur",
        "nom_du_site_de_comptage",
        "nom_du_compteur",
        "lien_vers_photo_du_site_de_comptage",
        "id_photos",
        "test_lien_vers_photos_du_site_de_comptage_",
        "id_photo_1",
        "url_sites",
        "type_dimage",
        "coordonnées_géographiques",
        "date_d'installation_du_site_de_comptage",
        "latitude",
        "longitude",
    ]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # feature list
    features = [c for c in df.columns if c not in ["comptage_horaire", "date_et_heure_de_comptage"]]

    return df, features


# ===========================================================
#  LIGHTGBM MODEL TRAINING PIPELINE
# ===========================================================
def train_final_model(X, y, model_params, target_cols, numeric_cols, test_size_ratio=0.1):
    # chronological split
    split_idx = int(len(X) * (1 - test_size_ratio))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    preprocessor = ColumnTransformer(
        transformers=[
            ("target_enc", TargetEncoder(), target_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    model = LGBMRegressor(**model_params)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }

    return pipeline, metrics


# ===========================================================
#  AUTO ARIMA OPTIMIZATION (optional)
# ===========================================================
def optimize_auto_arima(series):
    """
    NOTE: Requires pmdarima:
      pip install pmdarima
    """
    # model = pm.auto_arima(series, seasonal=True, m=24, stepwise=True, suppress_warnings=True)
    # return model
    raise NotImplementedError("auto_arima is disabled because pmdarima import is commented out.")


# ===========================================================
# SARIMAX BASELINE (optional)
# ===========================================================
def train_sarimax(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results
