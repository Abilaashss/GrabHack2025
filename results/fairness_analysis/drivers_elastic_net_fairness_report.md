
# Fairness Analysis Report
## Model: drivers_elastic_net
## Overall Fairness Score: 0.950

### Protected Attribute: GENDER

**Demographic Parity:**
- Difference: 0.0017
- Ratio: 0.9869

**Equalized Odds:**
- TPR Difference: 0.0333
- FPR Difference: 0.0000
- Average Difference: 0.0167

**Equal Opportunity:**
- Difference: 0.0333

**Individual Fairness:**
- Average Violation: 59.64
- Max Violation: 352.19

**Calibration:**
- Difference: 3.2373

### Protected Attribute: ETHNICITY

**Demographic Parity:**
- Difference: 0.0244
- Ratio: 0.8210

**Equalized Odds:**
- TPR Difference: 0.0530
- FPR Difference: 0.0000
- Average Difference: 0.0265

**Equal Opportunity:**
- Difference: 0.0530

**Individual Fairness:**
- Average Violation: 59.78
- Max Violation: 258.18

**Calibration:**
- Difference: 7.2257

### Protected Attribute: AGE_GROUP

**Demographic Parity:**
- Difference: 0.0162
- Ratio: 0.8798

**Equalized Odds:**
- TPR Difference: 0.0693
- FPR Difference: 0.0000
- Average Difference: 0.0347

**Equal Opportunity:**
- Difference: 0.0693

**Individual Fairness:**
- Average Violation: 59.99
- Max Violation: 258.18

**Calibration:**
- Difference: 5.6806

## Fairness Interpretation

✅ **Excellent fairness** - The model shows minimal bias across protected groups.
