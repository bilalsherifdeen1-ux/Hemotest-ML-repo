"""
HemoTest ML — NeuroVitalis / EQUIDX AI Biotech
================================================
Smartphone-based haemoglobin prediction for point-of-care anaemia screening.

Scientific basis:
  Cyanmethemoglobin (HiCN) colorimetric method — WHO/ICSH gold standard.
  Beer-Lambert Law: A = eps * c * l  (peak absorbance at 540 nm).
  OD_SLOPE = 0.0366 per g/dL  (ICSH HiCN empirical calibration constant).

WHO 2024 thresholds:
  WHO. Guideline on haemoglobin cutoffs to define anaemia. Geneva; 2024.
  Braat S et al. Lancet Haematol. 2024. doi:10.1016/S2352-3026(24)00030-9
"""
from hemotest.predictor import HemoTestPredictor
from hemotest.clinical.classification import AnemiaClassifier, AnemiaResult
from hemotest.features.colorimetry import ColorimetryExtractor

__version__ = "1.0.0"
__author__ = "Sherifdeen Bilal Olamilekan — NeuroVitalis / EQUIDX AI"
__email__ = "team@neurovitalis.ng"
__all__ = ["HemoTestPredictor", "AnemiaClassifier", "AnemiaResult", "ColorimetryExtractor"]
