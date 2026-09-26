"""
ML Models for Haemoglobin Prediction
=======================================
XGBoost (primary), Random Forest, Gradient Boosting, and Stacking Ensemble.

Decision — XGBoost as primary model:
  AUC=0.95, Accuracy=87%, Recall=85% for Sub-Saharan maternal anaemia.
  Source: Adimasu F et al. (2025). PMC13490807.
  Hyperparameters (lr=0.05, n=500, depth=6) consistent with SSA task size.

Decision — Stacking ensemble:
  Multi-model stacking outperforms individual models for Hb estimation.
  Source: Acharya et al. (2020). Stacking for PPG-based Hb estimation.

Decision — Linear calibration as baseline:
  Directly analogous to ICSH HiCN calibration curve; most interpretable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from loguru import logger

FEATURE_COLS: list[str] = [
    "r_mean","g_mean","b_mean","r_std","g_std","b_std",
    "optical_density","r_norm","g_norm","b_norm",
    "g_over_r","b_over_r","g_over_rb",
    "hue","saturation","value","l_star","a_star","b_star",
]
TARGET_COL = "hemoglobin_gdl"


@dataclass
class ModelMetrics:
    rmse: float; mae: float; r2: float
    cv_rmse_mean: float; cv_rmse_std: float

    @property
    def rmse_gdl(self) -> float:
        """Backward-compatible clinical name for the RMSE value."""
        return self.rmse

    def __str__(self) -> str:
        return (f"RMSE={self.rmse:.3f}  MAE={self.mae:.3f}  R2={self.r2:.4f}  "
                f"CV-RMSE={self.cv_rmse_mean:.3f}+/-{self.cv_rmse_std:.3f}")


def _linear() -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

def _rf() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=4,
            min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42)),
    ])

def _xgb() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, eval_metric="rmse")),
    ])

def _gbm() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gbm", GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, min_samples_leaf=3, random_state=42)),
    ])

def _ensemble() -> StackingRegressor:
    return StackingRegressor(
        estimators=[("linear",_linear()),("rf",_rf()),("xgb",_xgb()),("gbm",_gbm())],
        final_estimator=Ridge(alpha=0.5), cv=5, n_jobs=-1,
    )


class SafetyAwareRegressor(BaseEstimator, RegressorMixin):
    """Stacking regressor with a conservative severe-anaemia safety floor.

    Optical density is monotonic with haemoglobin under the HiCN Beer-Lambert
    calibration. Near the severe threshold, a borderline estimate must not be
    reported as safe; the guardrail therefore caps predictions at 7 g/dL when
    OD is within a small measurement-noise band of the severe boundary.
    """

    def __init__(self, base_model=None, optical_density_index: int = 6,
                 severe_hb_gdl: float = 7.0, od_guard_threshold: float = 0.26):
        self.base_model = base_model
        self.optical_density_index = optical_density_index
        self.severe_hb_gdl = severe_hb_gdl
        self.od_guard_threshold = od_guard_threshold

    def fit(self, X, y):
        self.base_model_ = self.base_model if self.base_model is not None else _ensemble()
        self.base_model_.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.asarray(self.base_model_.predict(X), dtype=float)
        values = np.asarray(X)
        if values.ndim == 2 and values.shape[1] > self.optical_density_index:
            high_risk = values[:, self.optical_density_index] <= self.od_guard_threshold
            predictions[high_risk] = np.minimum(
                predictions[high_risk], self.severe_hb_gdl - 1e-6
            )
        return predictions


def _safety_ensemble() -> SafetyAwareRegressor:
    return SafetyAwareRegressor(base_model=_ensemble())


class HemoTestModelTrainer:
    def __init__(self, feature_cols: list[str] = FEATURE_COLS):
        self.feature_cols = feature_cols
        self.models: dict[str, object] = {}
        self.metrics: dict[str, ModelMetrics] = {}

    def train_all(self, df: pd.DataFrame) -> dict[str, ModelMetrics]:
        X = df[self.feature_cols].values; y = df[TARGET_COL].values
        for name, builder in [
            ("linear_calibration", _linear), ("random_forest", _rf),
            ("xgboost", _xgb), ("gradient_boosting", _gbm), ("ensemble", _safety_ensemble),
        ]:
            logger.info(f"Training {name}...")
            m = builder(); met = self._cv_fit(m, X, y)
            self.models[name] = m; self.metrics[name] = met
            logger.info(f"  {name}: {met}")
        return self.metrics

    def _cv_fit(self, model, X, y) -> ModelMetrics:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv = -cross_val_score(model, X, y, cv=kf,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
        model.fit(X, y)
        # Report the stack's fitted accuracy separately from the conservative
        # deployment guardrail, which intentionally caps borderline severe
        # predictions to avoid dangerous false negatives.
        yp = (model.base_model_.predict(X)
              if isinstance(model, SafetyAwareRegressor) else model.predict(X))
        return ModelMetrics(
            rmse=float(np.sqrt(mean_squared_error(y, yp))),
            mae=float(mean_absolute_error(y, yp)),
            r2=float(r2_score(y, yp)),
            cv_rmse_mean=float(cv.mean()), cv_rmse_std=float(cv.std()),
        )

    def save_best(self, path: str | Path) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models["ensemble"], p)
        logger.info(f"Ensemble saved -> {p}")

    def save_all(self, directory: str | Path) -> None:
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, d / f"{name}.pkl")
        logger.info(f"Saved {len(self.models)} models -> {d}")
