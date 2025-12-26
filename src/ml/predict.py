from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# ---------- Paths ----------
DATA_PATH = Path("data/processed/ml_dataset.csv")
MODEL_PATH = Path("models/latest_model.joblib")
OUT_PATH = Path("results/ml/predictions.csv")

# Confidence Threshold (Only predict P if confidence > 55%)
THRESHOLD = 0.55

def main():
    print("Starting Prediction...")

    # 1. Load Data and Model
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        print("Error: Missing data or model. Run train.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # 2. Align Features with Training Data
    # Get the exact list of columns the model was trained on
    expected_features = model.feature_names_in_
    
    # Select ONLY those columns from the new data
    # This automatically drops 'datetime', 'y', and extra booleans that weren't used
    X = df[expected_features].copy()
    
    # Fill missing values if any (safety check)
    X = X.fillna(0)

    print(f"Generating predictions for {len(X)} rows...")

    # 3. Predict Probabilities
    # proba[:, 1] gives the probability of Class 1 (Price going UP)
    proba_up = model.predict_proba(X)[:, 1]

    # Apply Custom Threshold
    pred_custom = (proba_up >= THRESHOLD).astype(int)

    # 4. Create Results DataFrame
    results = df.copy()
    results["ml_proba"] = proba_up
    results["ml_pred"] = pred_custom
    results["ml_threshold"] = THRESHOLD

    # 5. Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_PATH, index=False)

    print("--------------------------------------------------")
    print(f"Predictions saved to: {OUT_PATH}")
    print("Sample Output:")
    print(results[["ml_proba", "ml_pred", "y"]].tail(5))
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()