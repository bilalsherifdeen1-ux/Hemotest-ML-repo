# Engineering Write-up: Building HemoTest ML

## From smartphone colorimetry to a safety-aware screening service

**Project:** HemoTest ML  
**Organization:** NeuroVitalis Health Innovations — EQUIDX AI Biotech  
**Author:** Sherifdeen Bilal Olamilekan  
**Repository:** [bilalsherifdeen1-ux/Hemotest-ML-repo](https://github.com/bilalsherifdeen1-ux/Hemotest-ML-repo)

> HemoTest ML is a prototype point-of-care screening system that estimates haemoglobin concentration from colorimetric test-strip measurements. It is designed to support trained health workers—not to replace laboratory confirmation or clinical judgment.

---

## 1. The engineering problem

Anaemia screening is often constrained by the availability of analysers, consumables, power, and trained laboratory personnel. The engineering goal behind HemoTest ML is to turn a controlled colorimetric reaction into a repeatable software pipeline that can run on low-cost imaging hardware and expose a simple prediction interface.

That goal creates three simultaneous constraints:

1. **Physical validity:** the model must remain grounded in the chemistry and optics of haemoglobin measurement.
2. **Operational simplicity:** a health worker should be able to move from an image or feature payload to a structured result through one interface.
3. **Clinical conservatism:** a prototype screening result must not silently classify a potentially life-threatening Hb value as safe.

The repository therefore treats the problem as more than a regression exercise. It implements a complete path from calibration data and feature extraction to model training, clinical classification, API validation, and automated safety tests.

---

## 2. System architecture

The system is organized as a layered pipeline:

```text
Image or feature payload
        |
        v
ColorimetryExtractor
        |
        |  19 optical and colour features
        v
HemoTestModelTrainer / ensemble
        |
        |  haemoglobin estimate in g/dL
        v
Safety-aware prediction boundary
        |
        v
WHO-oriented clinical classification
        |
        v
FastAPI response / Python result
```

The main code boundaries are deliberately narrow:

- [`src/hemotest/features/colorimetry.py`](../src/hemotest/features/colorimetry.py) owns image-to-feature conversion.
- [`src/hemotest/data/generator.py`](../src/hemotest/data/generator.py) creates reproducible calibration data for development and testing.
- [`src/hemotest/models/ml_models.py`](../src/hemotest/models/ml_models.py) owns model construction, cross-validation, persistence, and safety-aware inference.
- [`src/hemotest/clinical/classification.py`](../src/hemotest/clinical/classification.py) applies patient-context thresholds and severity bands.
- [`src/hemotest/predictor.py`](../src/hemotest/predictor.py) provides the end-to-end Python interface.
- [`api/app.py`](../api/app.py) exposes health, feature prediction, and image prediction routes.
- [`tests/`](../tests/) protects the scientific, clinical, model, and API contracts.

This separation makes it possible to improve one layer—for example, replace the image ROI detector—without changing the clinical classification API.

---

## 3. Translating the measurement physics into features

HemoTest uses the green RGB channel as a practical proxy for the 540 nm absorbance peak of the cyanmethemoglobin reaction. The optical-density feature is derived from the Beer–Lambert relationship:

```text
OD = -log10(G_sample / G_reference)
```

The implementation uses an empirical reference value of `G_reference = 200`, with the calibration assumption documented in [`docs/science.md`](science.md). The resulting feature is not treated as a black-box image embedding; it is an explicit measurement proxy that can be inspected, recalibrated, and quality-controlled.

The extractor returns 19 features in a stable order:

| Feature group | Features | Engineering purpose |
|---|---|---|
| Channel statistics | RGB means and standard deviations | Capture signal level and local variation |
| Optical density | Beer–Lambert green-channel OD | Encode the primary photometric relationship |
| Normalised RGB | `r_norm`, `g_norm`, `b_norm` | Reduce sensitivity to overall illumination |
| Channel ratios | `G/R`, `B/R`, `G/(R+B)` | Capture colour balance and reaction shifts |
| HSV | Hue, saturation, value | Represent intuitive colour and brightness changes |
| CIELab | `L*`, `a*`, `b*` | Add a perceptually structured colour space |

The feature vector is intentionally small. A compact, interpretable vector is easier to validate on constrained devices and easier to diagnose than an unbounded end-to-end vision model.

### ROI handling and numerical safeguards

The extractor supports an explicit region of interest and a lightweight automatic ROI detector. It also guards common image failure modes:

- empty images raise a clear `ValueError`;
- image channels are blurred before aggregate statistics are computed;
- divisions use small positive epsilons;
- optical density uses a minimum green-channel value to avoid invalid logarithms;
- downstream haemoglobin output is clipped to the supported range `[2.0, 22.0]` g/dL.

These choices are not substitutes for device calibration. They are defensive boundaries that keep malformed inputs from becoming undefined clinical outputs.

---

## 4. Reproducible data and model training

The repository includes a synthetic calibration generator so that the complete pipeline can be exercised without embedding patient data in the codebase. The generator samples haemoglobin values using a Nigeria-oriented distribution, converts them into colorimetric signals using the documented calibration constant, and adds controlled measurement noise.

The training command is:

```bash
python scripts/train.py --generate-data --n-samples 500
```

Training produces five persisted estimators:

- polynomial ridge baseline (`linear_calibration`);
- random forest regressor;
- XGBoost regressor;
- gradient-boosting regressor;
- stacked ensemble with a ridge meta-learner.

The trainer uses shuffled five-fold cross-validation and records RMSE, MAE, R², and cross-validated RMSE. Models are saved with `joblib` under `models/`, including the deployment artifact `models/ensemble_v1.pkl`.

A key design decision is to retain the linear calibration model as a baseline. It provides a physically interpretable reference point and prevents the ensemble from being evaluated only against an arbitrary metric target.

> **Benchmark qualification:** the repository’s generated-data metrics are software and pipeline checks. They are not evidence of clinical performance. Clinical claims require a locked protocol, representative real-world samples, an independent holdout set, device/site stratification, and prospective validation.

---

## 5. Safety-aware inference

A conventional regression objective rewards average accuracy. In a screening workflow, the cost of a dangerous severe-anaemia false negative is not symmetric with the cost of a small overestimate near the decision boundary.

HemoTest therefore wraps the stacked ensemble in a safety-aware regressor. Near the severe-anaemia boundary, the wrapper uses the monotonic optical-density feature as a conservative guardrail. When the measured OD is within the configured high-risk band, the deployment prediction is capped below `7.0 g/dL` rather than allowing a borderline estimate to be reported as non-severe.

This design has two important properties:

1. **The underlying model remains inspectable.** Fit metrics are calculated from the base ensemble, while deployment predictions retain the safety behavior.
2. **The safety rule is testable.** [`tests/test_models.py`](../tests/test_models.py) explicitly asserts zero severe false negatives on the reproducible training fixture.

The guardrail is a screening safety mechanism, not a clinical guarantee. Real validation must quantify its sensitivity, calibration impact, false-positive burden, and performance under device, lighting, reagent, and population shifts.

---

## 6. Clinical classification is separate from prediction

The model estimates haemoglobin. It does not decide treatment. The classification layer takes the estimate together with patient context—sex, pregnancy status, trimester, age, altitude, and smoking—and returns a structured result with severity and recommendation fields.

Keeping classification separate from regression prevents a common failure mode: mixing a changing clinical policy threshold into the learned model weights. Thresholds can therefore be reviewed, tested, and updated independently of model retraining.

The test suite covers the documented population-specific thresholds, including the updated second-trimester pregnancy threshold, as well as severity categories and altitude/smoking adjustments.

---

## 7. API and operational boundary

The FastAPI service exposes a small, typed surface:

- health/readiness information;
- prediction from a validated feature vector;
- prediction from an uploaded image;
- structured clinical context fields.

Pydantic schemas reject invalid sex values, malformed payloads, and feature vectors with the wrong length before they reach the model. This makes the API boundary a data-quality control rather than merely a transport layer.

Local development:

```bash
uvicorn api.app:app --reload
```

The generated OpenAPI documentation is available at `/docs`. The same package can also be used directly from Python through `HemoTestPredictor`, which is useful for batch evaluation and integration tests.

For a production deployment, the next operational controls should include authentication, encrypted transport, structured audit logs without unnecessary identifiers, model/version pinning, calibration checks, rate limiting, and a documented rollback path.

---

## 8. Verification strategy

The repository uses layered verification instead of relying on one aggregate score. The final local verification run passed **36 tests**:

- API health and validation tests;
- WHO-oriented threshold and severity tests;
- Beer–Lambert and 19-feature extraction tests;
- metric and severe-case safety tests;
- model performance and persistence tests.

The development workflow is:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

The project also includes a CI workflow so that dependency installation and tests remain part of the repository’s normal change path.

Warnings from third-party libraries—such as FastAPI lifecycle deprecations and joblib process warnings—are currently non-blocking. They should be cleaned up as part of a later dependency-modernisation pass, but they do not invalidate the current test results.

---

## 9. What is production-ready—and what is not

### Engineering foundations in place

- deterministic synthetic data generation;
- explicit, inspectable colour features;
- multiple model families and cross-validation;
- persisted model artifacts;
- typed API validation;
- clinical classification separated from regression;
- safety-oriented severe-case test coverage;
- Docker and CI scaffolding;
- MIT-licensed source repository.

### Remaining validation work

HemoTest ML remains a prototype. Before clinical deployment, the system needs:

1. a prospective, real-sample validation study with a laboratory reference method;
2. an independent holdout set from different operators, devices, lighting conditions, and sites;
3. subgroup analysis for pregnancy, age, sex, skin tone, haemoglobin variants, and co-morbidities;
4. calibration drift monitoring and a documented reference-card procedure;
5. measurement uncertainty and repeatability analysis;
6. usability, human-factors, and workflow studies at the intended point of care;
7. regulatory, quality-management, and cybersecurity review.

The current model card records these limitations and should be treated as part of the release artifact, not as optional supplementary documentation.

---

## 10. Engineering lessons

### 1. Start with a measurable signal

The strongest part of the design is the explicit optical-density feature. It gives the model a physically meaningful anchor and gives engineers a diagnostic quantity to inspect when predictions drift.

### 2. Make safety a code path, not a slogan

A statement such as “severe cases must not be missed” is not an engineering control until it exists as a test, a failure log, and a deployment rule. HemoTest encodes that requirement in both the model wrapper and the test suite.

### 3. Separate model quality from clinical policy

Regression metrics answer one question: how close is the estimate to the reference? Clinical thresholds answer another: how should a result be interpreted for this person? Keeping those concerns separate makes the system easier to review and maintain.

### 4. Treat synthetic data as scaffolding

Synthetic data is valuable for reproducibility, CI, and early integration. It cannot establish clinical validity. The project is intentionally explicit about that boundary so that benchmark performance is not mistaken for evidence of real-world effectiveness.

### 5. Ship the verification path with the feature

The repository is more useful because it includes training, evaluation, API tests, model persistence, documentation, and deployment scaffolding together. The test command is part of the product interface for future contributors.

---

## Conclusion

HemoTest ML demonstrates an end-to-end engineering pattern for a constrained clinical screening prototype: begin with a known measurement relationship, convert it into auditable features, compare several model families, isolate clinical policy from regression, add explicit safety behavior, and verify the public API and scientific contracts together.

The next milestone is not a larger model. It is disciplined real-world validation: representative samples, independent sites, transparent uncertainty, and a regulatory-quality evidence trail. That is the path from a reproducible repository to a trustworthy point-of-care system.
