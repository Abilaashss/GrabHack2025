
# Fairness Analysis Report
## Model: drivers_lasso
## Overall Fairness Score: 0.945

### Protected Attribute: GENDER

**Demographic Parity:**
- Difference: 0.0012
- Ratio: 0.9943

**Equalized Odds:**
- TPR Difference: 0.0424
- FPR Difference: 0.0000
- Average Difference: 0.0212

**Equal Opportunity:**
- Difference: 0.0424

**Individual Fairness:**
- Average Violation: 83.81
- Max Violation: 485.13

**Calibration:**
- Difference: 4.6893

### Protected Attribute: ETHNICITY

**Demographic Parity:**
- Difference: 0.0207
- Ratio: 0.9050

**Equalized Odds:**
- TPR Difference: 0.0162
- FPR Difference: 0.0000
- Average Difference: 0.0081

**Equal Opportunity:**
- Difference: 0.0162

**Individual Fairness:**
- Average Violation: 84.02
- Max Violation: 365.50

**Calibration:**
- Difference: 20.4618

### Protected Attribute: AGE_GROUP

**Demographic Parity:**
- Difference: 0.0171
- Ratio: 0.9221

**Equalized Odds:**
- TPR Difference: 0.0478
- FPR Difference: 0.0000
- Average Difference: 0.0239

**Equal Opportunity:**
- Difference: 0.0478

**Individual Fairness:**
- Average Violation: 84.29
- Max Violation: 365.50

**Calibration:**
- Difference: 2.5908

## Fairness Interpretation

✅ **Excellent fairness** - The model shows minimal bias across protected groups.
