"""
Colorimetric Feature Extraction
=================================
Extracts Beer-Lambert optical density and colour-space features from
test strip images for haemoglobin prediction.

Physics:
  540 nm = cyanmethemoglobin peak absorbance = green RGB channel.
  OD = -log10(G_sample / G_reference)   [Beer-Lambert]
  G_reference = 200  [Whatman Grade 1 paper under 540 nm LED, Ahsan 2023]

References:
  [1] Ahsan M et al. (2023). Sensors 23(1):394. doi:10.3390/s23010394
  [2] Mutlu AY et al. (2017). Analyst 142:2434. doi:10.1039/c7an00025a
  [3] Zhang Y et al. (2025). Science Advances. doi:10.1126/sciadv.adt4831
"""
from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple
from loguru import logger


@dataclass
class ColorFeatures:
    """19 colour-space features from a test strip image."""
    r_mean: float; g_mean: float; b_mean: float
    r_std: float;  g_std: float;  b_std: float
    optical_density: float
    r_norm: float; g_norm: float; b_norm: float
    g_over_r: float; b_over_r: float; g_over_rb: float
    hue: float; saturation: float; value: float
    l_star: float; a_star: float; b_star: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.r_mean, self.g_mean, self.b_mean,
            self.r_std,  self.g_std,  self.b_std,
            self.optical_density,
            self.r_norm, self.g_norm, self.b_norm,
            self.g_over_r, self.b_over_r, self.g_over_rb,
            self.hue, self.saturation, self.value,
            self.l_star, self.a_star, self.b_star,
        ], dtype=np.float32)

    FEATURE_NAMES: ClassVar[list[str]] = [
        "r_mean","g_mean","b_mean","r_std","g_std","b_std",
        "optical_density","r_norm","g_norm","b_norm",
        "g_over_r","b_over_r","g_over_rb",
        "hue","saturation","value","l_star","a_star","b_star",
    ]


class ColorimetryExtractor:
    """
    Extracts colorimetric features from the test strip reaction zone.

    Args:
        g_reference: Green channel value of blank (un-bloodied) strip.
                     Default 200 is empirical for Whatman Grade 1 filter paper
                     under 540 nm LED. Source: Ahsan et al. Sensors 2023 Fig 3.
                     Recalibrate weekly using compute_g_reference().
    """
    DEFAULT_G_REFERENCE = 200.0

    def __init__(
        self,
        g_reference: float = DEFAULT_G_REFERENCE,
        roi: Optional[Tuple[int, int, int, int]] = None,
        auto_detect_roi: bool = True,
    ):
        self.g_reference = g_reference
        self.roi = roi
        self.auto_detect_roi = auto_detect_roi

    def extract(self, image_bgr: np.ndarray) -> ColorFeatures:
        """Extract 19 colorimetric features from a test strip BGR image."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty or None image provided")

        roi = self._get_roi(image_bgr)
        roi = cv2.GaussianBlur(roi, (5, 5), 1)   # Reduce photon shot noise

        b_c, g_c, r_c = cv2.split(roi)
        rm = float(np.mean(r_c)); gm = float(np.mean(g_c)); bm = float(np.mean(b_c))
        rs = float(np.std(r_c));  gs = float(np.std(g_c));  bs = float(np.std(b_c))

        eps = 1e-6
        total = rm + gm + bm + eps
        od = float(-np.log10(max(gm, 1.0) / self.g_reference))   # Beer-Lambert

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        lc, ac, bc = cv2.split(lab)

        return ColorFeatures(
            r_mean=rm, g_mean=gm, b_mean=bm,
            r_std=rs,  g_std=gs,  b_std=bs,
            optical_density=od,
            r_norm=rm/total, g_norm=gm/total, b_norm=bm/total,
            g_over_r=gm/max(rm, eps),
            b_over_r=bm/max(rm, eps),
            g_over_rb=gm/max(rm+bm, eps),
            hue=float(np.mean(h))*2.0,
            saturation=float(np.mean(s))/255.0,
            value=float(np.mean(v))/255.0,
            l_star=float(np.mean(lc))*100.0/255.0,
            a_star=float(np.mean(ac))-128.0,
            b_star=float(np.mean(bc))-128.0,
        )

    def _get_roi(self, img: np.ndarray) -> np.ndarray:
        if self.roi:
            x, y, w, h = self.roi
            return img[y:y+h, x:x+w]
        if self.auto_detect_roi:
            return self._auto_detect(img)
        return img

    @staticmethod
    def _auto_detect(img: np.ndarray) -> np.ndarray:
        _, s, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        _, mask = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("Auto-detect ROI failed — using full image")
            return img
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        x=max(0,x-5); y=max(0,y-5)
        w=min(img.shape[1]-x,w+10); h=min(img.shape[0]-y,h+10)
        return img[y:y+h, x:x+w]

    @classmethod
    def compute_g_reference(cls, blank_images: list[np.ndarray]) -> float:
        """Compute G_reference from blank strips — run weekly (QC procedure)."""
        vals = [float(np.mean(cv2.split(i)[1])) for i in blank_images]
        ref = float(np.mean(vals))
        logger.info(f"G_reference = {ref:.2f} from {len(blank_images)} blank strips")
        return ref
