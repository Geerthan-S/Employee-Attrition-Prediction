# Model Card — Employee Attrition Prediction

## Overview

This model is part of an academic machine-learning mini project that predicts the probability of employee attrition. It is **not intended for real-world automated employment decisions**.

## Final Model

- **Estimator:** Logistic Regression
- **Hyperparameter tuning:** 5-fold stratified GridSearchCV
- **Best `C`:** 5
- **Class weight:** None in the tuned final estimator
- **Decision threshold:** 0.27 (selected on validation data by maximizing F1)
- **Data split:** 70% train / 15% validation / 15% test
- **Protected feature:** `Gender` excluded from predictors

## Final Untouched Test Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.8235 |
| Precision | 0.4717 |
| Recall | 0.6944 |
| F1 | 0.5618 |
| ROC-AUC | 0.8333 |
| PR-AUC | 0.6365 |

The model achieved stronger recall than the default 0.50-threshold configuration because the threshold was optimized on the validation set before test evaluation.

## Explainability

Permutation importance on the held-out test set identified the following as the highest-impact original features for this trained model:

- `OverTime` — importance 0.1323
- `JobRole` — importance 0.0441
- `YearsWithCurrManager` — importance 0.0329
- `BusinessTravel` — importance 0.0292
- `JobInvolvement` — importance 0.0249
- `MonthlyIncome` — importance 0.0248
- `Age` — importance 0.0211
- `DistanceFromHome` — importance 0.0195

These are **model associations, not causal effects**. The repository also includes a SHAP global-importance visualization.

## Fairness Diagnostic

`Gender` is excluded from the predictor set. It is used only for a small post-hoc subgroup diagnostic on the test set. The subgroup sample sizes are limited, so this is **not a complete fairness audit**.

| Group   |   Samples |   Positive Cases |   Accuracy |   Precision |   Recall |     F1 |
|:--------|----------:|-----------------:|-----------:|------------:|---------:|-------:|
| Female  |        84 |               12 |     0.7857 |      0.375  |   0.75   | 0.5    |
| Male    |       137 |               24 |     0.8467 |      0.5517 |   0.6667 | 0.6038 |

## Limitations

- Small educational dataset (1,470 records).
- Historical patterns may not generalize to other organizations or time periods.
- The model may reflect biases or structural patterns in the source data.
- Threshold optimization is specific to this dataset and objective.
- Explainability describes model behavior, not causality.
- A real deployment would require governance, privacy, legal, fairness, drift, and human-oversight controls.

## Intended Use

**Permitted intended use:** classroom demonstration of an end-to-end ML workflow.

**Not intended:** hiring, firing, promotion, compensation, disciplinary action, or automated retention decisions about real employees.
