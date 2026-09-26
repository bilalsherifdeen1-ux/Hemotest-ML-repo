"""Evaluate a saved model on test data."""
import typer, joblib, pandas as pd, numpy as np
from pathlib import Path
from hemotest.evaluation.metrics import evaluate_model
from hemotest.models.ml_models import FEATURE_COLS, TARGET_COL

def main(
    model: Path = typer.Option(Path("models/ensemble_v1.pkl")),
    data: Path = typer.Option(Path("data/processed/calibration_data.csv")),
    anemia_threshold: float = typer.Option(12.0),
):
    m = joblib.load(model)
    df = pd.read_csv(data)
    y_pred = np.clip(m.predict(df[FEATURE_COLS].values), 2.0, 22.0)
    print(evaluate_model(df[TARGET_COL].values, y_pred, anemia_threshold).summary())

if __name__ == "__main__":
    typer.run(main)
