"""
Clinical Evaluation Metrics
=============================
Bland-Altman agreement analysis and diagnostic performance metrics.

References:
  [1] Bland JM, Altman DG (1986). Lancet 1(8476):307.
  [2] ICSH (2001). Guidelines for POC testing. Haematologica 86(5).
      Minimum r>0.90 acceptable; r>0.95 excellent for POC Hb.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_auc_score,
)
from scipy import stats
from loguru import logger


@dataclass
class ClinicalValidationReport:
    pearson_r: float; pearson_p: float; r_squared: float
    rmse_gdl: float; mae_gdl: float; bias_gdl: float
    loa_upper: float; loa_lower: float
    within_1gdl_pct: float; within_2gdl_pct: float
    sensitivity_pct: float; specificity_pct: float
    ppv_pct: float; npv_pct: float; auc: float
    severe_sensitivity_pct: float; severe_false_neg_n: int
    n_samples: int; anemia_threshold_gdl: float

    def summary(self) -> str:
        safe = "PASS" if self.severe_false_neg_n==0 else f"FAIL ({self.severe_false_neg_n} missed)"
        return (
            f"\nHEMOTEST CLINICAL VALIDATION (n={self.n_samples})\n"
            f"  Pearson r={self.pearson_r:.4f}  R2={self.r_squared:.4f}\n"
            f"  RMSE={self.rmse_gdl:.3f}  MAE={self.mae_gdl:.3f} g/dL\n"
            f"  Bias={self.bias_gdl:+.3f}  LoA=[{self.loa_lower:.3f}, {self.loa_upper:.3f}]\n"
            f"  Within +-1 g/dL: {self.within_1gdl_pct:.1f}%\n"
            f"  Sensitivity={self.sensitivity_pct:.1f}%  Specificity={self.specificity_pct:.1f}%\n"
            f"  AUC={self.auc:.4f}\n"
            f"  Severe anaemia safety (Hb<7.0): {safe}\n"
        )


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anemia_threshold: float = 12.0,
) -> ClinicalValidationReport:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    r, p = stats.pearsonr(y_true, y_pred)
    diff = y_pred - y_true; bias = float(np.mean(diff)); sd = float(np.std(diff, ddof=1))
    ta = (y_true < anemia_threshold).astype(int)
    pa = (y_pred < anemia_threshold).astype(int)
    tn,fp,fn,tp = confusion_matrix(ta, pa).ravel()
    try:    auc = float(roc_auc_score(ta, -y_pred))
    except: auc = float("nan")
    ts = y_true < 7.0; ps = y_pred < 7.0
    sev_fn = int(ts.sum() - (ts & ps).sum())
    if sev_fn > 0:
        logger.error(f"SAFETY: {sev_fn} severe anaemia case(s) missed — NOT clinically deployable")
    return ClinicalValidationReport(
        pearson_r=float(r), pearson_p=float(p), r_squared=float(r2_score(y_true,y_pred)),
        rmse_gdl=float(np.sqrt(mean_squared_error(y_true,y_pred))),
        mae_gdl=float(mean_absolute_error(y_true,y_pred)),
        bias_gdl=bias, loa_upper=bias+1.96*sd, loa_lower=bias-1.96*sd,
        within_1gdl_pct=float((np.abs(diff)<=1.0).mean()*100),
        within_2gdl_pct=float((np.abs(diff)<=2.0).mean()*100),
        sensitivity_pct=tp/max(tp+fn,1)*100, specificity_pct=tn/max(tn+fp,1)*100,
        ppv_pct=tp/max(tp+fp,1)*100, npv_pct=tn/max(tn+fn,1)*100,
        auc=auc, severe_sensitivity_pct=float((ts&ps).sum()/max(ts.sum(),1)*100),
        severe_false_neg_n=sev_fn, n_samples=len(y_true),
        anemia_threshold_gdl=anemia_threshold,
    )
