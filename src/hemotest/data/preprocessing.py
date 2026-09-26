"""Data preprocessing — cleaning and validating calibration datasets."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split as _split
from loguru import logger

FEATURE_COLS = [
    "r_mean","g_mean","b_mean","r_std","g_std","b_std",
    "optical_density","r_norm","g_norm","b_norm",
    "g_over_r","b_over_r","g_over_rb",
    "hue","saturation","value","l_star","a_star","b_star",
]
TARGET_COL = "hemoglobin_gdl"


def preprocess_features(
    df: pd.DataFrame,
    clip_hb: bool = True,
    remove_outliers: bool = True,
    iqr_factor: float = 3.0,
) -> pd.DataFrame:
    n = len(df)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].clip(lower=0.0)
    if clip_hb:
        df[TARGET_COL] = df[TARGET_COL].clip(2.0, 22.0)
    df = df[df["optical_density"] >= 0].copy()
    rgb_s = df["r_norm"] + df["g_norm"] + df["b_norm"]
    df = df[rgb_s.between(0.9, 1.1)].copy()
    if remove_outliers:
        for col in ["optical_density", TARGET_COL]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[df[col].between(q1 - iqr_factor*iqr, q3 + iqr_factor*iqr)]
    df = df.dropna(subset=FEATURE_COLS+[TARGET_COL]).reset_index(drop=True)
    logger.info(f"Preprocessing: {n} -> {len(df)} rows ({n-len(df)} removed)")
    return df


def stratified_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = (df[TARGET_COL] < 12.0).astype(int)
    tr, te = _split(df, test_size=test_size, random_state=random_state, stratify=strata)
    return tr.reset_index(drop=True), te.reset_index(drop=True)
