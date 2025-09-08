
# Fairness Analysis Report
## Model: drivers_linear_regression
## Overall Fairness Score: 0.945

### Protected Attribute: GENDER

**Demographic Parity:**
- Difference: 0.0052
- Ratio: 0.9764

**Equalized Odds:**
- TPR Difference: 0.0308
- FPR Difference: 0.0000
- Average Difference: 0.0154

**Equal Opportunity:**
- Difference: 0.0308

**Individual Fairness:**
- Average Violation: 85.68
- Max Violation: 494.87

**Calibration:**
- Difference: 4.7585

### Protected Attribute: ETHNICITY

**Demographic Parity:**
- Difference: 0.0212
- Ratio: 0.9049

**Equalized Odds:**
- TPR Difference: 0.0174
- FPR Difference: 0.0000
- Average Difference: 0.0087

**Equal Opportunity:**
- Difference: 0.0174

**Individual Fairness:**
- Average Violation: 85.89
- Max Violation: 374.01

**Calibration:**
- Difference: 21.1428

### Protected Attribute: AGE_GROUP

**Demographic Parity:**
- Difference: 0.0171
- Ratio: 0.9239

**Equalized Odds:**
- TPR Difference: 0.0480
- FPR Difference: 0.0000
- Average Difference: 0.0240

**Equal Opportunity:**
- Difference: 0.0480

**Individual Fairness:**
- Average Violation: 86.17
- Max Violation: 374.01

**Calibration:**
- Difference: 2.8180

## Fairness Interpretation

✅ **Excellent fairness** - The model shows minimal bias across protected groups.
