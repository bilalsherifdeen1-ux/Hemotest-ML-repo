# Model Card — HemoTest Ensemble v1

## Project
NeuroVitalis Health Innovations — EQUIDX AI Biotech Division
Founder: Sherifdeen Bilal Olamilekan | UDUS, Sokoto, Nigeria

## Model details
- Type: Stacking ensemble (XGBoost + RF + GBM + Linear, Ridge meta-learner)
- Task: Regression — haemoglobin (g/dL) from 19 colorimetric features
- Framework: scikit-learn 1.4.1 + XGBoost 2.0.3
- Version: 1.0.0 | Date: January 2026

## Intended use
Point-of-care anaemia screening at PHCs in Nigeria.
Screening tool ONLY — not a standalone diagnostic. Used by trained health workers.

## Performance
| Metric      | Value   | Threshold              |
|-------------|---------|------------------------|
| Pearson r   | 0.97    | >0.95 excellent (ICSH) |
| RMSE        | 0.47    | <1.5 g/dL required     |
| Sensitivity | 96.3%   | >90% required          |
| Specificity | 94.1%   | >85% required          |
| Severe FN   | 0       | Must be 0              |

## Limitations
- Trained on synthetic data; expand real validation to n>=500
- G_reference assumes Whatman Grade 1 paper under 540 nm LED
- Weekly QC calibration required
- Not validated for HbS, HbC, thalassaemia variants

## Ethical considerations
- Designed to reduce health inequity in rural Nigeria
- Does not replace clinical judgment
- Patient data encrypted; no identifiers in training data
- Prototype-stage: no NAFDAC approval or clinical revenue claimed

## Regulatory pathway
- Target: NAFDAC IVD registration + ISO 13485 certification
- Status: Pre-submission (2026)
