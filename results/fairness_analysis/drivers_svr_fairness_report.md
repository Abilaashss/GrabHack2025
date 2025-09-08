
# Fairness Analysis Report
## Model: drivers_svr
## Overall Fairness Score: 0.949

### Protected Attribute: GENDER

**Demographic Parity:**
- Difference: 0.0063
- Ratio: 0.9709

**Equalized Odds:**
- TPR Difference: 0.0220
- FPR Difference: 0.0017
- Average Difference: 0.0119

**Equal Opportunity:**
- Difference: 0.0220

**Individual Fairness:**
- Average Violation: 84.97
- Max Violation: 350.81

**Calibration:**
- Difference: 2.4453

### Protected Attribute: ETHNICITY

**Demographic Parity:**
- Difference: 0.0220
- Ratio: 0.9003

**Equalized Odds:**
- TPR Difference: 0.0161
- FPR Difference: 0.0011
- Average Difference: 0.0086

**Equal Opportunity:**
- Difference: 0.0161

**Individual Fairness:**
- Average Violation: 85.23
- Max Violation: 331.74

**Calibration:**
- Difference: 2.2666

### Protected Attribute: AGE_GROUP

**Demographic Parity:**
- Difference: 0.0136
- Ratio: 0.9377

**Equalized Odds:**
- TPR Difference: 0.0309
- FPR Difference: 0.0035
- Average Difference: 0.0172

**Equal Opportunity:**
- Difference: 0.0309

**Individual Fairness:**
- Average Violation: 85.49
- Max Violation: 312.36

**Calibration:**
- Difference: 2.0301

## Fairness Interpretation

✅ **Excellent fairness** - The model shows minimal bias across protected groups.
