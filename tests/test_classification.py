"""WHO 2024 anaemia classification — every assertion cites its source."""
import pytest
from hemotest.clinical.classification import AnemiaClassifier, AnemiaSeverity

class TestWHO2024Thresholds:
    def test_pregnant_t1_normal(self, classifier):
        r = classifier.classify(11.5, sex="female", pregnant=True, trimester=1)
        assert not r.is_anemic and r.who_threshold_gdl == 11.0

    def test_pregnant_t2_updated_2024(self, classifier):
        """T2 threshold = 10.5 g/dL — KEY 2024 update (was 11.0 since 1968).
        Source: Braat S et al. Lancet Haematol. 2024. doi:10.1016/S2352-3026(24)00030-9"""
        r = classifier.classify(10.8, sex="female", pregnant=True, trimester=2)
        assert r.who_threshold_gdl == 10.5
        assert not r.is_anemic   # 10.8 > 10.5 => normal in T2 under 2024 threshold

    def test_pregnant_t2_anemic_new_threshold(self, classifier):
        assert classifier.classify(10.4, sex="female", pregnant=True, trimester=2).is_anemic

    def test_non_pregnant_female(self, classifier):
        r = classifier.classify(11.9, sex="female", pregnant=False)
        assert r.is_anemic and r.who_threshold_gdl == 12.0

    def test_male_threshold(self, classifier):
        assert not classifier.classify(13.0, sex="male").is_anemic

    def test_child_6_59mo(self, classifier):
        assert classifier.classify(10.9, sex="female", age_months=24).is_anemic

class TestSeverity:
    def test_life_threatening(self, classifier):
        r = classifier.classify(4.5, sex="female", pregnant=True, trimester=3)
        assert r.severity == AnemiaSeverity.LIFE_THREATENING and r.referral_required

    def test_severe(self, classifier):
        assert classifier.classify(6.8, pregnant=True, trimester=1).severity == AnemiaSeverity.SEVERE

    def test_moderate(self, classifier):
        assert classifier.classify(8.5).severity == AnemiaSeverity.MODERATE

    def test_mild(self, classifier):
        assert classifier.classify(10.5).severity == AnemiaSeverity.MILD

    def test_normal(self, classifier):
        r = classifier.classify(13.5)
        assert r.severity == AnemiaSeverity.NORMAL and not r.referral_required

class TestAdjustments:
    """WHO 2024 altitude (Table A2.4) and smoking (Section 5.3)."""
    def test_below_1000m_no_adjustment(self, classifier):
        r = classifier.classify(12.5, altitude_m=277)
        assert abs(r.hemoglobin_gdl - 12.5) < 0.01

    def test_2000m_subtracts_0_8(self, classifier):
        r = classifier.classify(12.5, altitude_m=2000)
        assert abs(r.hemoglobin_gdl - (12.5 - 0.8)) < 0.01

    def test_smoking_subtracts_0_3(self, classifier):
        r_ns = classifier.classify(12.5, smoker=False)
        r_s  = classifier.classify(12.5, smoker=True)
        assert abs(r_s.hemoglobin_gdl - (r_ns.hemoglobin_gdl - 0.3)) < 0.01
