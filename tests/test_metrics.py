"""Clinical metrics validation tests."""
import pytest, numpy as np
from hemotest.evaluation.metrics import evaluate_model

def test_zero_bias_perfect_prediction():
    y = np.array([8.0,10.0,12.0,14.0,6.5])
    r = evaluate_model(y, y)
    assert abs(r.bias_gdl) < 0.01 and abs(r.loa_upper) < 0.01

def test_perfect_pearson():
    y = np.linspace(4.0, 18.0, 50)
    assert evaluate_model(y, y).pearson_r == pytest.approx(1.0, abs=1e-6)

def test_severe_miss_detected():
    y_true = np.array([6.0, 8.0, 12.0, 14.0])
    y_pred = np.array([8.0, 8.0, 12.0, 14.0])   # severe case at 6.0 missed
    assert evaluate_model(y_true, y_pred).severe_false_neg_n == 1

def test_perfect_sensitivity_specificity():
    y = np.array([8.0,10.0,13.0,14.0])
    r = evaluate_model(y, y)
    assert r.sensitivity_pct == pytest.approx(100.0)
    assert r.specificity_pct == pytest.approx(100.0)
