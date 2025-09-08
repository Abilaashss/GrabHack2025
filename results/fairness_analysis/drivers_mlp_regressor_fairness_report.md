
# Fairness Analysis Report
## Model: drivers_mlp_regressor
## Overall Fairness Score: 0.954

### Protected Attribute: GENDER

**Demographic Parity:**
- Difference: 0.0136
- Ratio: 0.9474

**Equalized Odds:**
- TPR Difference: 0.0032
- FPR Difference: 0.0000
- Average Difference: 0.0016

**Equal Opportunity:**
- Difference: 0.0032

**Individual Fairness:**
- Average Violation: 87.61
- Max Violation: 371.20

**Calibration:**
- Difference: 0.0606

### Protected Attribute: ETHNICITY

**Demographic Parity:**
- Difference: 0.0242
- Ratio: 0.9056

**Equalized Odds:**
- TPR Difference: 0.0005
- FPR Difference: 0.0000
- Average Difference: 0.0002

**Equal Opportunity:**
- Difference: 0.0005

**Individual Fairness:**
- Average Violation: 87.85
- Max Violation: 357.05

**Calibration:**
- Difference: 0.0809

### Protected Attribute: AGE_GROUP

**Demographic Parity:**
- Difference: 0.0137
- Ratio: 0.9464

**Equalized Odds:**
- TPR Difference: 0.0093
- FPR Difference: 0.0000
- Average Difference: 0.0047

**Equal Opportunity:**
- Difference: 0.0093

**Individual Fairness:**
- Average Violation: 88.19
- Max Violation: 331.48

**Calibration:**
- Difference: 0.0813

## Fairness Interpretation

✅ **Excellent fairness** - The model shows minimal bias across protected groups.
