# Scientific Documentation — HemoTest ML

## Cyanmethemoglobin (HiCN) Method

    Hb (Fe2+) + K3Fe(CN)6  ->  MetHb (Fe3+)
    MetHb (Fe3+) + KCN      ->  HiCN  (stable red-brown complex)

Peak absorbance at 540 nm (green spectrum).

## Beer-Lambert Law

    A = eps * c * l
    eps(HiCN, 540 nm) = 11,000 L/mol/cm   [ICSH 1978]
    MW(Hb per haem) = 16,114 g/mol
    l = 1 cm (fixed optical path in HemoTest device)
    => c(g/dL) -> A = (11000 / (16114 * 10)) * c = 0.0366 * c

OD_SLOPE = 0.0366 per g/dL  (core calibration constant)

Green channel as 540 nm proxy:
  OD = -log10(G_sample / G_reference)
  G_reference = 200  (Whatman Grade 1 paper, 540 nm LED — Ahsan 2023)

## WHO 2024 Thresholds

| Group                   | Threshold  | Notes              |
|-------------------------|------------|--------------------|
| Pregnant T1 & T3        | < 11.0 g/dL| Unchanged 1968     |
| Pregnant T2             | < 10.5 g/dL| UPDATED 2024       |
| Non-pregnant women      | < 12.0 g/dL| Unchanged          |
| Men >= 15 yr            | < 13.0 g/dL| Unchanged          |
| Children 6-59 months    | < 11.0 g/dL| Reaffirmed 2024    |

Reference: WHO (2024). Guideline on haemoglobin cutoffs. CC BY-NC-SA 3.0 IGO.

## Nigeria Epidemiology

- Pregnant women anaemia prevalence: 62-68%
  Source: Obio-Akpor study 2019-2023 (n=2,290), PMC12908788
- PHCs without Hb testing: ~98%  (FMOH 2022)
- Maternal mortality ratio: 1,047/100,000 (WHO 2020)

## References

1. WHO (2024). Guideline on haemoglobin cutoffs. Geneva.
2. ICSH (1978). Reference method for haemoglobinometry. J Clin Pathol 31:139.
3. Ahsan M et al. (2023). Sensors 23(1):394. doi:10.3390/s23010394
4. Braat S et al. (2024). Lancet Haematol. doi:10.1016/S2352-3026(24)00030-9
5. Adimasu F et al. (2025). XGBoost AUC=0.95. PMC13490807.
6. Mutlu AY et al. (2017). Analyst 142:2434.
7. Bland JM, Altman DG (1986). Lancet 1(8476):307.
