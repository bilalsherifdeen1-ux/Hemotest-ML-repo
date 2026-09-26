# Data Directory

data/
├── raw/            Real blood sample measurements (git-ignored; never commit PII)
└── processed/      Preprocessed datasets for training (git-ignored)

## Calibration protocol
Use certified Hb calibration solutions covering 0-18 g/dL in 2 g/dL steps.
Run each standard in triplicate. Record RGB image + device OD reading.
Validate against ICSH HiCN reference: OD_slope = 0.0366 per g/dL.
