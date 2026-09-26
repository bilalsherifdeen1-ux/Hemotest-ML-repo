"""ML model training and safety tests."""
import numpy as np
import pytest
from hemotest.models.ml_models import FEATURE_COLS, TARGET_COL
from hemotest.evaluation.metrics import evaluate_model

class TestPerformance:
    def test_xgboost_r2_above_0_90(self, trained_trainer):
        assert trained_trainer.metrics["xgboost"].r2 >= 0.90

    def test_xgboost_rmse_below_1_5(self, trained_trainer):
        assert trained_trainer.metrics["xgboost"].rmse_gdl < 1.5

    def test_ensemble_beats_linear(self, trained_trainer):
        assert (trained_trainer.metrics["ensemble"].rmse_gdl <
                trained_trainer.metrics["linear_calibration"].rmse_gdl)

    def test_all_models_trained(self, trained_trainer):
        assert set(trained_trainer.models) == {
            "linear_calibration","random_forest","xgboost","gradient_boosting","ensemble"}

class TestClinicalSafety:
    """SAFETY: severe anaemia (Hb < 7.0) must NEVER be missed."""
    def test_zero_severe_false_negatives(self, training_data, trained_trainer):
        X = training_data[FEATURE_COLS].values
        y = training_data[TARGET_COL].values
        yp = trained_trainer.models["ensemble"].predict(X)
        r = evaluate_model(y, yp)
        assert r.severe_false_neg_n == 0, (
            f"SAFETY FAIL: {r.severe_false_neg_n} life-threatening cases missed.")

    def test_sensitivity_above_90(self, training_data, trained_trainer):
        X = training_data[FEATURE_COLS].values
        y = training_data[TARGET_COL].values
        yp = trained_trainer.models["ensemble"].predict(X)
        assert evaluate_model(y, yp).sensitivity_pct >= 90.0

    def test_within_2gdl_above_95pct(self, training_data, trained_trainer):
        X = training_data[FEATURE_COLS].values
        y = training_data[TARGET_COL].values
        yp = trained_trainer.models["ensemble"].predict(X)
        assert (np.abs(yp-y)<=2.0).mean()*100 >= 95.0
