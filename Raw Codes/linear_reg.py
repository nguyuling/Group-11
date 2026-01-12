"""Simple and multiple linear regression on the cleaned leukemia dataset.

Run directly:
	python linear_reg.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURES_PATH = "GSE13164_cleaned_features.csv"
LABELS_PATH = "GSE13164_cleaned_labels.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_aligned_data(features_path: str, labels_path: str):
	"""Load features/labels and align on Sample_ID."""
	features = pd.read_csv(features_path, index_col=0)
	labels = pd.read_csv(labels_path)

	combined = labels.set_index("Sample_ID").join(features, how="inner")

	# Prefer numeric target if available
	if "Target_Code" in combined.columns:
		y = combined["Target_Code"].astype(float)
	else:
		y = combined["Leukemia_Type"].astype("category").cat.codes.astype(float)

	# Drop label columns to get feature matrix
	drop_cols = [c for c in ["Leukemia_Type", "Target_Code"] if c in combined.columns]
	X = combined.drop(columns=drop_cols)

	return X, y


def best_single_feature(X: pd.DataFrame, y: pd.Series) -> str:
	"""Pick the feature with highest absolute Pearson correlation to target."""
	corrs = X.apply(lambda col: np.corrcoef(col, y)[0, 1])
	best_feature = corrs.abs().idxmax()
	return best_feature


def train_and_eval(name: str, X_train, X_test, y_train, y_test):
	model = Pipeline([
		("scaler", StandardScaler()),
		("linreg", LinearRegression()),
	])

	model.fit(X_train, y_train)
	pred_train = model.predict(X_train)
	pred_test = model.predict(X_test)

	metrics = {
		"name": name,
		"train_r2": r2_score(y_train, pred_train),
		"test_r2": r2_score(y_test, pred_test),
		"train_mse": mean_squared_error(y_train, pred_train),
		"test_mse": mean_squared_error(y_test, pred_test),
	}

	print(f"\n{name}")
	print("-" * 60)
	print(f"Train R^2: {metrics['train_r2']:.4f} | Test R^2: {metrics['test_r2']:.4f}")
	print(f"Train MSE: {metrics['train_mse']:.4f} | Test MSE: {metrics['test_mse']:.4f}")

	return metrics


def main():
	X, y = load_aligned_data(FEATURES_PATH, LABELS_PATH)
	print(f"Loaded data: X shape {X.shape}, y length {len(y)}")

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
	)

	results = []

	# Simple linear regression on best single gene
	single_gene = best_single_feature(X_train, y_train)
	results.append(
		train_and_eval(
			f"Simple Linear Regression on {single_gene}",
			X_train[[single_gene]],
			X_test[[single_gene]],
			y_train,
			y_test,
		)
	)

	# Multiple linear regression on all genes
	results.append(
		train_and_eval(
			"Multiple Linear Regression (all genes)",
			X_train,
			X_test,
			y_train,
			y_test,
		)
	)

	# Summary
	print("\nSummary (sorted by test R^2):")
	for res in sorted(results, key=lambda r: r["test_r2"], reverse=True):
		print(
			f"{res['name']}: Test R^2={res['test_r2']:.4f}, Test MSE={res['test_mse']:.4f}"
		)


if __name__ == "__main__":
	main()
