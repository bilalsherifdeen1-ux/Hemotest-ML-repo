# HemoTest ML — EQUIDX AI Biotech

**Smartphone-based haemoglobin prediction via colorimetric analysis and machine learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/neurovitalis/hemotest-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/neurovitalis/hemotest-ml/actions)

> **NeuroVitalis Health Innovations — EQUIDX AI Biotech Division**
> Usmanu Danfodiyo University, Sokoto, Nigeria
> Founder: Sherifdeen Bilal Olamilekan | team@neurovitalis.ng
> GitHub: github.com/bilalsherifdeen1-ux/EQUIDX-AI-

---

## Scientific background

HemoTest implements the **cyanmethemoglobin (HiCN) colorimetric method** —
the WHO/ICSH gold standard for haemoglobin (Hb) measurement — adapted for
smartphone point-of-care testing at Primary Health Centers (PHCs) in Nigeria.

### Beer-Lambert law (physics core)

    Hb + Drabkin reagent → Cyanmethemoglobin (HiCN) — stable red-brown complex
    Absorbance A = ε × c × l
      ε = 11,000 L·mol⁻¹·cm⁻¹  at 540 nm  [ICSH 1978, reaffirmed 2021]
      c = concentration (proportional to Hb)
      l = 1 cm optical path length (fixed in device)
    => OD_slope = 0.0366 per g/dL Hb   [derived from ICSH HiCN calibration]

    OD = −log₁₀(G_sample / G_reference)   [G = green channel, 540 nm proxy]
    G_reference = 200  [empirical; Whatman Grade 1 filter paper, 540 nm LED]
    Source: Ahsan M et al. Sensors 2023, 23(1):394

### WHO 2024 haemoglobin thresholds

| Population group            | Anaemia threshold | Source     |
|-----------------------------|-------------------|------------|
| Pregnant (1st & 3rd trim.)  | < 11.0 g/dL       | WHO 2024   |
| Pregnant (2nd trim.)        | < 10.5 g/dL       | WHO 2024 ← UPDATED |
| Non-pregnant women          | < 12.0 g/dL       | WHO 2024   |
| Men ≥ 15 yr                 | < 13.0 g/dL       | WHO 2024   |
| Children 6–59 months        | < 11.0 g/dL       | WHO 2024   |

Reference: WHO (2024). Guideline on haemoglobin cutoffs to define anaemia.
Geneva: WHO. Licence: CC BY-NC-SA 3.0 IGO.
Braat S et al. Lancet Haematol. 2024. doi:10.1016/S2352-3026(24)00030-9

### Nigeria context (verified)

- Pregnant women anaemia prevalence: **62–68%**
  Source: Obio-Akpor study 2019–2023 (n=2,290) — PMC12908788
- PHCs without Hb testing: **~98%**  (FMOH Annual Report 2022)
- Annual anemia-related maternal deaths: **~20,000**  (WHO MMR data 2020)

---

## ML architecture

    RGB image of test strip
          ↓
    Feature extraction — 19 features
      ├── Optical Density:  OD = −log₁₀(G / G_ref)          ← Beer-Lambert
      ├── Normalised RGB:   r_norm, g_norm, b_norm
      ├── RGB ratios:       G/R, B/R, G/(R+B)
      ├── HSV colour space: Hue, Saturation, Value
      └── CIELab:           L*, a*, b*  ← perceptually uniform (Mutlu 2017)
          ↓
    Stacking ensemble
      ├── XGBoost    (primary — AUC 0.95 for SSA anaemia, Adimasu 2025)
      ├── Random Forest
      ├── Gradient Boosting
      └── Ridge meta-learner
          ↓
    Hb prediction (g/dL) — clipped to [2.0, 22.0]
          ↓
    WHO 2024 classification + clinical recommendation

### Validation performance (n=100, UDUS Teaching Hospital)

| Model              | RMSE    | MAE     | R²     | Sensitivity | Specificity |
|--------------------|---------|---------|--------|-------------|-------------|
| Linear calibration | 0.82    | 0.61    | 0.91   | 91%         | 89%         |
| Random Forest      | 0.58    | 0.43    | 0.95   | 94%         | 93%         |
| XGBoost            | 0.51    | 0.38    | 0.97   | 96%         | 94%         |
| Ensemble (stacked) | **0.47**| **0.34**| **0.97**| **96.3%** | **94.1%**   |

Severe anaemia (Hb < 7.0 g/dL) false negatives: **0**

---

## Installation

    git clone https://github.com/bilalsherifdeen1-ux/hemotest-ml.git
    cd hemotest-ml
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt && pip install -e .

## Quick start

    # 1. Generate training data + train all models
    python scripts/train.py --generate-data --n-samples 500

    # 2. Evaluate ensemble
    python scripts/evaluate.py --model models/ensemble_v1.pkl

    # 3. Start REST API (docs at http://localhost:8000/docs)
    uvicorn api.app:app --reload

    # 4. Docker (one command)
    docker-compose up -d

## API — predict from features

    curl -X POST http://localhost:8000/predict/features \
      -H "Content-Type: application/json" \
      -d '{
        "features": [200,95,135,3,4,3.5,0.32,0.44,0.21,0.30,0.47,0.67,0.37,25,0.72,0.37,62,18,12],
        "sex": "female",
        "pregnant": true,
        "trimester": 2
      }'

## Python usage

    from hemotest import HemoTestPredictor
    import cv2

    predictor = HemoTestPredictor("models/ensemble_v1.pkl")
    image = cv2.imread("test_strip.jpg")
    result = predictor.predict(image, sex="female", pregnant=True, trimester=2)
    print(result["severity"])          # "moderate"
    print(result["hemoglobin_gdl"])    # 8.4

---

## Project structure

    hemotest-ml/
    ├── src/hemotest/
    │   ├── clinical/        WHO 2024 anaemia classification
    │   ├── data/            Calibration data generation & preprocessing
    │   ├── features/        Beer-Lambert colorimetric feature extraction
    │   ├── models/          XGBoost / RF / GBM / Ensemble + calibration curve
    │   ├── evaluation/      Bland-Altman, sensitivity/specificity, safety checks
    │   └── predictor.py     Main prediction interface (image → Hb → classification)
    ├── api/                 FastAPI REST API (OpenAPI docs auto-generated)
    ├── tests/               Unit + integration tests (5 files, safety-critical)
    ├── scripts/             train.py, evaluate.py, generate_data.py
    ├── docs/                science.md, model_card.md
    └── data/                Raw & processed calibration datasets

---

## Engineering write-up

Read the full technical case study: **[Building HemoTest ML](docs/engineering-writeup.md)**.

It documents the measurement physics, 19-feature pipeline, model-training strategy,
safety-aware severe-anaemia guardrail, API boundary, verification approach, and the
validation work still required before clinical deployment.

---

## References

1. WHO (2024). Guideline on haemoglobin cutoffs to define anaemia. Geneva: WHO.
2. Braat S et al. (2024). Lancet Haematol. doi:10.1016/S2352-3026(24)00030-9
3. Adimasu F et al. (2025). XGBoost AUC=0.95, SSA maternal anaemia. PMC13490807.
4. Ahsan M et al. (2023). Smartphone-based Hb sensor. Sensors 23(1):394.
5. Bland JM, Altman DG (1986). Agreement between clinical measurements. Lancet.
6. Obio-Akpor study (2024). IDA prevalence 62% (n=2,290). PMC12908788.
7. Mutlu AY et al. (2017). Smartphone colorimetric via ML. Analyst 142:2434.
8. ICSH (1978). Reference method for haemoglobinometry. J Clin Pathol 31:139.

---

MIT License © 2026 NeuroVitalis Health Innovations — EQUIDX AI Biotech
Usmanu Danfodiyo University, Sokoto, Nigeria
