"""
WHO 2024 Anaemia Classification
================================
Implements updated 2024 WHO haemoglobin cutoffs and severity grading.

Key 2024 change vs 1968 thresholds:
  Pregnant 2nd trimester: 11.0 → 10.5 g/dL.
  Source: Braat S et al. Lancet Haematol. 2024. doi:10.1016/S2352-3026(24)00030-9

Altitude adjustments from: WHO 2024 Guideline, Table A2.4.
Smoking adjustment from:   WHO 2024 Guideline, Section 5.3 (subtract 0.3 g/dL).

FIGO (2025) severity grading used: doi:10.1002/ijgo.70529
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from loguru import logger


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"


class Trimester(int, Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3


class AnemiaSeverity(str, Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life_threatening"   # Hb < 5.0 g/dL


@dataclass
class AnemiaResult:
    hemoglobin_gdl: float
    severity: AnemiaSeverity
    who_threshold_gdl: float
    is_anemic: bool
    sex: Sex
    pregnant: bool
    trimester: Optional[Trimester]
    recommendation: str
    referral_required: bool
    treatment: list[str] = field(default_factory=list)
    monitoring: str = ""
    color_code: str = ""

    def to_dict(self) -> dict:
        return {
            "hemoglobin_gdl": round(self.hemoglobin_gdl, 1),
            "severity": self.severity.value,
            "who_threshold_gdl": self.who_threshold_gdl,
            "is_anemic": self.is_anemic,
            "sex": self.sex.value,
            "pregnant": self.pregnant,
            "trimester": self.trimester.value if self.trimester else None,
            "recommendation": self.recommendation,
            "referral_required": self.referral_required,
            "treatment": self.treatment,
            "monitoring": self.monitoring,
            "color_code": self.color_code,
        }


class AnemiaClassifier:
    """
    WHO 2024 anaemia classifier.

    Usage:
        clf = AnemiaClassifier()
        result = clf.classify(8.4, sex="female", pregnant=True, trimester=2)
        result.severity   # AnemiaSeverity.MODERATE
    """

    # WHO 2024 haemoglobin thresholds (g/dL) — Table 1
    THRESHOLDS: dict[str, float] = {
        "pregnant_t1": 11.0,
        "pregnant_t2": 10.5,   # KEY 2024 UPDATE: was 11.0 since 1968
        "pregnant_t3": 11.0,
        "non_pregnant_female": 12.0,
        "male_adult": 13.0,
        "child_6_23mo": 11.0,
        "child_24_59mo": 11.0,
        "child_5_11yr": 11.5,
        "child_12_14yr": 12.0,
    }

    # WHO 2024 altitude adjustment — g/dL to SUBTRACT from measured Hb
    # Source: WHO 2024 Guideline, Table A2.4
    ALTITUDE_ADJ: dict[int, float] = {
        1000: 0.2, 1500: 0.5, 2000: 0.8, 2500: 1.3,
        3000: 1.9, 3500: 2.7, 4000: 3.5, 4500: 4.5,
    }

    def classify(
        self,
        hb: float,
        sex: str | Sex = "female",
        pregnant: bool = False,
        trimester: int | Trimester | None = None,
        age_months: int | None = None,
        altitude_m: float = 0.0,
        smoker: bool = False,
    ) -> AnemiaResult:
        """Classify anaemia status using WHO 2024 guidelines."""
        if isinstance(sex, str):
            sex = Sex(sex)
        if isinstance(trimester, int):
            trimester = Trimester(trimester)

        hb_adj = self._altitude_adjust(hb, altitude_m)
        if smoker:
            hb_adj -= 0.3          # WHO 2024 Section 5.3

        threshold = self._threshold(sex, pregnant, trimester, age_months)
        severity = self._severity(hb_adj, pregnant)
        referral = severity in (AnemiaSeverity.SEVERE, AnemiaSeverity.LIFE_THREATENING)
        rec, treatment, monitoring = self._recommendations(severity, pregnant)

        return AnemiaResult(
            hemoglobin_gdl=hb_adj,
            severity=severity,
            who_threshold_gdl=threshold,
            is_anemic=hb_adj < threshold,
            sex=sex,
            pregnant=pregnant,
            trimester=trimester,
            recommendation=rec,
            referral_required=referral,
            treatment=treatment,
            monitoring=monitoring,
            color_code=self._color(severity),
        )

    def _threshold(
        self,
        sex: Sex,
        pregnant: bool,
        trimester: Optional[Trimester],
        age_months: Optional[int],
    ) -> float:
        if age_months is not None:
            if age_months < 24:  return self.THRESHOLDS["child_6_23mo"]
            if age_months < 60:  return self.THRESHOLDS["child_24_59mo"]
            if age_months < 132: return self.THRESHOLDS["child_5_11yr"]
            return self.THRESHOLDS["child_12_14yr"]
        if pregnant:
            if trimester is None:
                logger.warning("Trimester not specified — using T1/T3 threshold 11.0")
                return self.THRESHOLDS["pregnant_t1"]
            return {
                Trimester.FIRST:  self.THRESHOLDS["pregnant_t1"],
                Trimester.SECOND: self.THRESHOLDS["pregnant_t2"],
                Trimester.THIRD:  self.THRESHOLDS["pregnant_t3"],
            }[trimester]
        return self.THRESHOLDS["male_adult" if sex == Sex.MALE else "non_pregnant_female"]

    def _severity(self, hb: float, pregnant: bool) -> AnemiaSeverity:
        # Severity breakpoints — WHO 2024; FIGO 2025 doi:10.1002/ijgo.70529
        if hb < 5.0:  return AnemiaSeverity.LIFE_THREATENING
        if hb < 7.0:  return AnemiaSeverity.SEVERE
        if hb < 10.0: return AnemiaSeverity.MODERATE
        if hb < (11.0 if pregnant else 12.0): return AnemiaSeverity.MILD
        return AnemiaSeverity.NORMAL

    def _altitude_adjust(self, hb: float, altitude_m: float) -> float:
        """WHO 2024 Table A2.4 — subtract adjustment from measured Hb."""
        if altitude_m < 1000:
            return hb
        alts = sorted(self.ALTITUDE_ADJ)
        if altitude_m >= alts[-1]:
            return hb - self.ALTITUDE_ADJ[alts[-1]]
        for i, a in enumerate(alts[:-1]):
            if a <= altitude_m < alts[i + 1]:
                frac = (altitude_m - a) / (alts[i + 1] - a)
                adj = self.ALTITUDE_ADJ[a] + frac * (
                    self.ALTITUDE_ADJ[alts[i + 1]] - self.ALTITUDE_ADJ[a])
                return hb - adj
        return hb

    def _recommendations(
        self, sev: AnemiaSeverity, pregnant: bool
    ) -> tuple[str, list[str], str]:
        pg = " (pregnancy increases risk)" if pregnant else ""
        if sev == AnemiaSeverity.NORMAL:
            return (
                "Haemoglobin within normal range. Continue routine care.",
                ["Iron-rich diet", "Prenatal vitamins if pregnant"],
                "Routine ANC" if pregnant else "Annual screening",
            )
        if sev == AnemiaSeverity.MILD:
            return (
                f"Mild anaemia{pg}. Start iron supplementation immediately.",
                [
                    "Ferrous sulfate 60 mg elemental iron once daily",
                    "Take with Vitamin C — increases absorption 3x",
                    "Avoid tea/coffee 1 hour before and after dose",
                    "Iron-rich diet: liver, beans, spinach, moringa",
                ],
                "Retest in 8 weeks. Refer if no improvement.",
            )
        if sev == AnemiaSeverity.MODERATE:
            return (
                f"Moderate anaemia{pg}. Urgent treatment required.",
                [
                    "Ferrous sulfate 60 mg TWICE daily",
                    "Folic acid 5 mg daily (essential if pregnant)",
                    "Vitamin C with every dose",
                    "High-iron diet: liver 2x/week, beans daily, moringa powder",
                    "Deworming if indicated (avoid albendazole in T1)",
                ],
                "Retest in 4 weeks. REFER immediately if Hb < 7.0 or no improvement.",
            )
        if sev == AnemiaSeverity.SEVERE:
            return (
                f"SEVERE ANAEMIA{pg}. IMMEDIATE hospital referral.",
                [
                    "REFER TO HOSPITAL NOW",
                    "IV iron infusion may be required (hospital only)",
                    "Blood transfusion if Hb < 6.0 or symptomatic",
                    "Investigate cause: malaria, hookworm, bleeding",
                ],
                "Hospital management. Do not manage at PHC level alone.",
            )
        return (
            "LIFE-THREATENING ANAEMIA. Ambulance NOW. Emergency transfusion.",
            ["CALL AMBULANCE IMMEDIATELY", "Oxygen if available", "IV access + normal saline"],
            "Immediate ICU/HDU transfer.",
        )

    @staticmethod
    def _color(sev: AnemiaSeverity) -> str:
        return {
            AnemiaSeverity.NORMAL: "green",
            AnemiaSeverity.MILD: "yellow",
            AnemiaSeverity.MODERATE: "orange",
            AnemiaSeverity.SEVERE: "red",
            AnemiaSeverity.LIFE_THREATENING: "dark_red",
        }[sev]
