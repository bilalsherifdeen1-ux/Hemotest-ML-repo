"""Shared pytest fixtures."""
import pytest, numpy as np, pandas as pd
from hemotest.clinical.classification import AnemiaClassifier
from hemotest.data.generator import CalibrationDataGenerator
from hemotest.models.ml_models import HemoTestModelTrainer, FEATURE_COLS

@pytest.fixture(scope="session")
def classifier() -> AnemiaClassifier:
    return AnemiaClassifier()

@pytest.fixture(scope="session")
def training_data() -> pd.DataFrame:
    return CalibrationDataGenerator(random_state=42).generate_calibration_dataset(
        n_samples=400, nigeria_distribution=True)

@pytest.fixture(scope="session")
def trained_trainer(training_data):
    t = HemoTestModelTrainer(feature_cols=FEATURE_COLS)
    t.train_all(training_data)
    return t

def make_image(r=200, g=120, b=140, size=60):
    img = np.zeros((size,size,3), dtype=np.uint8)
    img[:,:,0]=b; img[:,:,1]=g; img[:,:,2]=r
    return img
