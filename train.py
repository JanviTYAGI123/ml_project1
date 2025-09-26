import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: Path) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(f"Dataset not found at {csv_path}")
	return pd.read_csv(csv_path)


def build_pipeline(categorical_cols, numeric_cols) -> Pipeline:
	preprocess = ColumnTransformer(
		transformers=[
			("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
			("num", "passthrough", numeric_cols),
		]
	)
	regressor = RandomForestRegressor(n_estimators=300, random_state=42)
	model = Pipeline(steps=[("preprocess", preprocess), ("regressor", regressor)])
	return model


def train(csv_path: Path, out_path: Path):
	df = load_dataset(csv_path)

	# Expecting common insurance dataset columns
	required_cols = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
	missing = required_cols.difference(df.columns.str.lower())
	# If headers are capitalized/mixed, normalize
	df.columns = [c.lower() for c in df.columns]
	missing = required_cols.difference(df.columns)
	if missing:
		raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

	X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
	y = df["charges"]

	categorical_cols = ["sex", "smoker", "region"]
	numeric_cols = ["age", "bmi", "children"]
	model = build_pipeline(categorical_cols, numeric_cols)

	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
	model.fit(X_train, y_train)

	# Evaluate
	preds = model.predict(X_val)
	mae = mean_absolute_error(y_val, preds)
	r2 = r2_score(y_val, preds)
	print(f"Validation MAE: {mae:,.2f}")
	print(f"Validation R^2: {r2:.3f}")

	out_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, out_path, compress=3)
	print(f"Saved model to {out_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train insurance charges model and save joblib pipeline")
	parser.add_argument("--csv", type=str, default="insurance (1).csv", help="Path to insurance CSV")
	parser.add_argument("--out", type=str, default=str(Path("models") / "model.joblib"), help="Output model path")
	args = parser.parse_args()

	train(Path(args.csv), Path(args.out))


