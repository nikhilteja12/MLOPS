import argparse
import json
import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from my_utils import preprocess_data, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to raw CSV to evaluate on")
    parser.add_argument("--model", default="artifacts/model.pkl", help="Path to trained model.pkl")
    parser.add_argument("--out", default="artifacts/eval_metrics.json", help="Where to save eval metrics")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = load_model(args.model)

    df_raw = pd.read_csv(args.data)
    df_encoded, features = preprocess_data(df_raw)

    if "comptage_horaire" not in df_encoded.columns:
        raise ValueError("Cannot evaluate: 'comptage_horaire' not found after preprocessing.")

    X = df_encoded[features]
    y = df_encoded["comptage_horaire"]

    preds = model.predict(X)

    metrics = {
        "MAE": float(mean_absolute_error(y, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
        "R2": float(r2_score(y, preds)),
        "n_rows": int(len(df_encoded)),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(" Evaluation done.")
    print("Saved:", args.out)
    print(metrics)


if __name__ == "__main__":
    main()
