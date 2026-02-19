import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# ---------- Paths ----------
DATA_PATH = Path("data/processed/ml_dataset.csv")
MODEL_PATH = Path("models/latest_model.joblib")
METRICS_PATH = Path("results/ml/metrics.json")

def main():
    print("Starting Model Training...")

    # 1. Load Data
    if not DATA_PATH.exists():
        print(f"Error: Data not found at {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    
    # 2. Separate Features (X) and Target (y)
    # We drop 'y' (target) and 'future_ret' (used for calc, not training)
    y = df["y"].astype(int)
    
    # Drop known non-feature columns
    X = df.drop(columns=["y", "future_ret", "datetime"], errors="ignore")
    
    
    X = X.select_dtypes(include=['number'])

    # Clean infinite values and align Y
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    print(f"Data loaded: {len(X)} rows, {len(X.columns)} numeric features.")

    if X.empty:
        print("Error: No numeric features found. Check dataset.")
        return

    # 3. Setup Time-Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)

    # 4. Define Model (Random Forest)
    model = RandomForestClassifier(
        n_estimators=200,      
        max_depth=10,          
        min_samples_leaf=10,   
        random_state=42,       
        n_jobs=-1              
    )

    # 5. Train and Validate
    folds = []
    print("\n--- Cross-Validation Results ---")
    
    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        
        # Predictions
        pred = model.predict(X_test)
        
        # Safe probability calculation
        try:
            proba = model.predict_proba(X_test)[:, 1]
        except:
            proba = [0] * len(pred)

        # Calculate Scores
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, proba)
        except:
            auc = 0.5 

        print(f"Fold {fold_i}: Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        
        folds.append({
            "fold": fold_i,
            "accuracy": float(acc),
            "f1": float(f1),
            "roc_auc": float(auc)
        })

    # 6. Final Training on ALL Data
    print("\nTraining final model on full dataset...")
    model.fit(X, y)

    # 7. Save Model and Metrics
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    avg_acc = sum(f["accuracy"] for f in folds) / len(folds)
    avg_f1 = sum(f["f1"] for f in folds) / len(folds)
    
    metrics_payload = {
        "model": "RandomForestClassifier",
        "n_features": int(X.shape[1]),
        "features_list": list(X.columns),
        "cv_results": folds,
        "mean_accuracy": avg_acc,
        "mean_f1": avg_f1
    }
    
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2))

    print("--------------------------------------------------")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Average Accuracy: {avg_acc:.2%}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()