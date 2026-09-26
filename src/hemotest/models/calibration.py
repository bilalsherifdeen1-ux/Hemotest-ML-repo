"""
Beer-Lambert Calibration Curve
================================
Direct implementation of ICSH HiCN reference calibration.

Reference:
  ICSH (1978). J Clin Pathol 31(2):139.
  OD_slope = 0.0366 per g/dL at 540 nm.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.stats import linregress
from loguru import logger

ICSH_OD_SLOPE: float = 0.0366    # Per g/dL — ICSH HiCN reference
ICSH_OD_INTERCEPT: float = 0.0


@dataclass
class CalibrationCurve:
    slope: float; intercept: float
    r_squared: float; n_standards: int

    def predict(self, od: float | np.ndarray) -> float | np.ndarray:
        return (np.asarray(od) - self.intercept) / self.slope

    def od_from_hb(self, hb: float | np.ndarray) -> float | np.ndarray:
        return np.asarray(hb) * self.slope + self.intercept

    def __str__(self) -> str:
        return (f"Hb = (OD - {self.intercept:.4f}) / {self.slope:.4f}  "
                f"R2={self.r_squared:.4f}  n={self.n_standards}")


def fit_calibration_curve(
    hb_standards: np.ndarray, od_measurements: np.ndarray,
) -> CalibrationCurve:
    """Fit linear calibration from standard solutions (min 4 points)."""
    if len(hb_standards) < 4:
        raise ValueError("Minimum 4 calibration standards required.")
    slope, intercept, r, _, _ = linregress(hb_standards, od_measurements)
    curve = CalibrationCurve(float(slope), float(intercept), float(r**2), len(hb_standards))
    logger.info(f"Calibration: {curve}")
    if abs(slope - ICSH_OD_SLOPE) / ICSH_OD_SLOPE > 0.15:
        logger.warning(
            f"Slope {slope:.4f} deviates >15% from ICSH reference {ICSH_OD_SLOPE}. "
            "Check reagent prep and LED wavelength."
        )
    return curve

def icsh_reference() -> CalibrationCurve:
    return CalibrationCurve(ICSH_OD_SLOPE, ICSH_OD_INTERCEPT, 1.0, 0)
