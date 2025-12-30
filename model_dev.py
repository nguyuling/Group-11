import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
FEATURES_PATH = "GSE13164_cleaned_features.csv"
LABELS_PATH = "GSE13164_cleaned_labels.csv"
TOP_VAR_FEATURES_FOR_POLY = 30  # limit size for polynomial regression
RANDOM_STATE = 42


def load_and_prepare():
    """Load cleaned features/labels, align on Sample_ID, and return X, y, meta."""
    features_df = pd.read_csv(FEATURES_PATH, index_col=0)
    labels_df = pd.read_csv(LABELS_PATH)

    # Align to shared samples
    combined = labels_df.set_index("Sample_ID").join(features_df, how="inner")

    # Choose target: prefer existing numeric Target_Code; else encode Leukemia_Type
    if "Target_Code" in combined.columns:
        y = combined["Target_Code"].astype(float)
    else:
        # Simple encoding fallback
        y = combined["Leukemia_Type"].astype("category").cat.codes.astype(float)

    # Drop non-feature columns for X
    meta_cols = [c for c in ["Leukemia_Type", "Target_Code"] if c in combined.columns]
    X = combined.drop(columns=meta_cols)

    return X, y, combined


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    print(f"\n{name}")
    print("-" * 60)
    print(f"Train R^2: {train_r2:.4f} | Test R^2: {test_r2:.4f}")
    print(f"Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

    return {
        "name": name,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "y_test": y_test,
        "test_pred": test_pred,
    }


def plot_predictions(result):
    y_test = result["y_test"]
    test_pred = result["test_pred"]

    plt.figure(figsize=(10, 4))

    # Predicted vs True
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, test_pred, alpha=0.6)
    min_y, max_y = y_test.min(), y_test.max()
    plt.plot([min_y, max_y], [min_y, max_y], "r--", label="Ideal")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{result['name']}: True vs Pred")
    plt.legend()

    # Residuals
    plt.subplot(1, 2, 2)
    residuals = test_pred - y_test
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(f"{result['name']}: Residuals")

    plt.tight_layout()
    plt.show()


def main():
    X, y, combined = load_and_prepare()
    print(f"Loaded aligned data: X shape {X.shape}, y length {len(y)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []

    # ----------------------------------------------------------------
    # 1) Simple Linear Regression (single feature)
    # ----------------------------------------------------------------
    first_gene = X.columns[0]
    simple_model = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression()),
    ])
    res_simple = evaluate_model(
        f"Simple Linear Regression on {first_gene}",
        simple_model,
        X_train[[first_gene]],
        y_train,
        X_test[[first_gene]],
        y_test,
    )
    results.append(res_simple)

    # ----------------------------------------------------------------
    # 2) Multiple Linear Regression (all genes, scaled)
    # ----------------------------------------------------------------
    multi_model = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression()),
    ])
    res_multi = evaluate_model(
        "Multiple Linear Regression (all genes)",
        multi_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    results.append(res_multi)

    # ----------------------------------------------------------------
    # 3) Polynomial Regression (top-variance genes to keep it tractable)
    # ----------------------------------------------------------------
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(TOP_VAR_FEATURES_FOR_POLY).index.tolist()
    poly_model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("linreg", LinearRegression()),
    ])
    res_poly = evaluate_model(
        f"Polynomial Regression (degree=2, top {len(top_features)} genes)",
        poly_model,
        X_train[top_features],
        y_train,
        X_test[top_features],
        y_test,
    )
    results.append(res_poly)

    # Show metrics summary
    print("\nSummary of models:")
    for r in results:
        print(
            f"{r['name']}: Test R^2={r['test_r2']:.4f}, Test MSE={r['test_mse']:.4f}"
        )

    # Visualize predictions for the best test R^2 model
    best = max(results, key=lambda r: r["test_r2"])
    print(f"\nBest model by Test R^2: {best['name']}")
    plot_predictions(best)

    # Decision making / prediction example
    sample_preds = best["test_pred"][:5]
    print("\nSample predictions (first 5 of test set):")
    for i, pred in enumerate(sample_preds, 1):
        print(f"Sample {i}: predicted target {pred:.3f}")


if __name__ == "__main__":
    main()