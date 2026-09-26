"""Beer-Lambert law and feature extraction tests."""
import math, pytest, numpy as np
from tests.conftest import make_image
from hemotest.features.colorimetry import ColorimetryExtractor, ColorFeatures

@pytest.fixture
def ext():
    return ColorimetryExtractor(g_reference=200.0)

class TestBeerLambert:
    def test_blank_od_near_zero(self, ext):
        assert abs(ext.extract(make_image(g=200)).optical_density) < 0.02

    def test_od_increases_lower_g(self, ext):
        assert ext.extract(make_image(g=80)).optical_density > ext.extract(make_image(g=180)).optical_density

    def test_od_formula_numeric(self, ext):
        """OD = -log10(100/200) = log10(2) = 0.301"""
        assert abs(ext.extract(make_image(g=100)).optical_density - math.log10(2)) < 0.03

    def test_clinical_od_range(self, ext):
        """Hb~12: OD=0.0366*12=0.439 => G=200*10^(-0.439)=72.7"""
        f = ext.extract(make_image(g=73))
        assert 0.35 < f.optical_density < 0.55

class TestFeatures:
    def test_19_features(self, ext):
        assert ext.extract(make_image()).to_array().shape == (19,)

    def test_normalised_rgb_sums_to_1(self, ext):
        f = ext.extract(make_image(r=100,g=150,b=80))
        assert abs(f.r_norm+f.g_norm+f.b_norm-1.0) < 0.001

    def test_empty_raises(self, ext):
        with pytest.raises(ValueError): ext.extract(np.array([]))
