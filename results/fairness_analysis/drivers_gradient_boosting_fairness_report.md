
# Fairness Analysis Report
## Model: drivers_gradient_boosting
## Overall Fairness Score: 0.948

### Protected Attribute: GENDER

**Demographic Parity:**
- Difference: 0.0023
- Ratio: 0.9888

**Equalized Odds:**
- TPR Difference: 0.0371
- FPR Difference: 0.0015
- Average Difference: 0.0193

**Equal Opportunity:**
- Difference: 0.0371

**Individual Fairness:**
- Average Violation: 78.42
- Max Violation: 322.43

**Calibration:**
- Difference: 1.5506

### Protected Attribute: ETHNICITY

**Demographic Parity:**
- Difference: 0.0190
- Ratio: 0.9096

**Equalized Odds:**
- TPR Difference: 0.0208
- FPR Difference: 0.0016
- Average Difference: 0.0112

**Equal Opportunity:**
- Difference: 0.0208

**Individual Fairness:**
- Average Violation: 78.67
- Max Violation: 292.74

**Calibration:**
- Difference: 3.5990

### Protected Attribute: AGE_GROUP

**Demographic Parity:**
- Difference: 0.0155
- Ratio: 0.9260

**Equalized Odds:**
- TPR Difference: 0.0479
- FPR Difference: 0.0014
- Average Difference: 0.0247

**Equal Opportunity:**
- Difference: 0.0479

**Individual Fairness:**
- Average Violation: 78.91
- Max Violation: 312.40

**Calibration:**
- Difference: 2.3454

## Fairness Interpretation

✅ **Excellent fairness** - The model shows minimal bias across protected groups.
