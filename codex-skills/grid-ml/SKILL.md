---
name: grid-ml
description: Build reproducible machine-learning systems for forecasting, fault detection, anomaly detection, classification, and grid decision support.
---

# Grid Machine Learning

## Use this skill when

Use this skill for power-system forecasting, event classification, fault detection, anomaly detection, ranking, feature engineering, model evaluation, explainability, or deployment of grid ML models.

## Required workflow

1. Define the operational or planning decision the model supports.
2. Define target, prediction horizon, unit of observation, and success metric.
3. Establish a simple baseline before training advanced models.
4. Split data chronologically when time dependence exists.
5. Prevent target, temporal, and entity leakage.
6. Build preprocessing as a reproducible pipeline.
7. Tune only using training and validation data.
8. Evaluate on an untouched test period.
9. Analyze errors by operating regime.
10. Document limitations and deployment assumptions.

## Data rules

- Preserve timestamps and timezone information.
- Document missing-data treatment.
- Do not randomly split time series unless justified.
- Avoid using future measurements in historical features.
- Track source, vintage, and revision of labels.
- Check class balance and rare-event coverage.

## Baselines

Use relevant baselines such as:

- persistence
- seasonal naive
- linear or logistic regression
- historical mean or median
- simple tree model

Advanced performance is meaningful only relative to an appropriate baseline.

## Metrics

For forecasting, consider MAE, RMSE, quantile loss, calibration, and regime-specific errors.

For classification, consider precision, recall, F1, PR-AUC, ROC-AUC, confusion matrices, and class-specific performance.

Do not rely only on accuracy for imbalanced problems.

## Explainability

Use feature importance, SHAP, partial dependence, residual analysis, and representative error cases where appropriate. Do not claim causal relationships from predictive explanations.

## Reproducibility

Record:

- dataset version
- feature list
- split dates
- preprocessing
- model parameters
- random seeds
- environment
- metrics
- artifact locations

## Completion format

Report:

- Problem framing
- Data and leakage controls
- Baseline
- Model results
- Error analysis
- Deployment risks
