
# Fairness Analysis Report
## Model: drivers_random_forest
## Overall Fairness Score: 0.946

### Protected Attribute: GENDER

**Demographic Parity:**
- Difference: 0.0020
- Ratio: 0.9913

**Equalized Odds:**
- TPR Difference: 0.0437
- FPR Difference: 0.0012
- Average Difference: 0.0224

**Equal Opportunity:**
- Difference: 0.0437

**Individual Fairness:**
- Average Violation: 81.43
- Max Violation: 339.57

**Calibration:**
- Difference: 1.3494

### Protected Attribute: ETHNICITY

**Demographic Parity:**
- Difference: 0.0199
- Ratio: 0.9144

**Equalized Odds:**
- TPR Difference: 0.0242
- FPR Difference: 0.0021
- Average Difference: 0.0132

**Equal Opportunity:**
- Difference: 0.0242

**Individual Fairness:**
- Average Violation: 81.63
- Max Violation: 308.06

**Calibration:**
- Difference: 0.4996

### Protected Attribute: AGE_GROUP

**Demographic Parity:**
- Difference: 0.0085
- Ratio: 0.9627

**Equalized Odds:**
- TPR Difference: 0.0473
- FPR Difference: 0.0053
- Average Difference: 0.0263

**Equal Opportunity:**
- Difference: 0.0473

**Individual Fairness:**
- Average Violation: 81.89
- Max Violation: 322.30

**Calibration:**
- Difference: 1.2035

## Fairness Interpretation

✅ **Excellent fairness** - The model shows minimal bias across protected groups.
