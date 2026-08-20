# Employee Attrition Prediction & Explainability 🚀

An advanced but compact machine-learning mini project using the IBM HR Analytics Employee Attrition dataset.

## Highlights

- EDA + data-quality audit
- leakage-safe **70/15/15 train-validation-test split**
- **4 ML algorithms**: Logistic Regression, Decision Tree, Random Forest, Extra Trees
- **5-fold stratified cross-validation**
- `GridSearchCV` hyperparameter tuning
- class-imbalance-aware metrics
- validation-only **decision-threshold optimization**
- Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC
- ROC + Precision-Recall curves
- permutation importance
- **SHAP explainability**
- basic fairness diagnostic with `Gender` excluded from model features
- saved preprocessing + model pipeline

## Verified Reference Run

The project was tested on the IBM dataset. The final candidate was **Tuned Logistic Regression** with a validation-selected threshold of **0.27**.

| Metric | Test score |
|---|---:|
| Accuracy | 0.8235 |
| Precision | 0.4717 |
| Recall | 0.6944 |
| F1 | 0.5618 |
| ROC-AUC | 0.8333 |
| PR-AUC | 0.6365 |

> These scores are for this fixed educational split and should not be interpreted as real-world HR performance.

## Repository Structure

```text
Employee-Attrition-Prediction/
├── Employee_Attrition_Prediction.ipynb
├── README.md
├── MODEL_CARD.md
├── requirements.txt
├── run_project.py
└── models/
    └── attrition_model.joblib
```

The notebook and script automatically download the public IBM HR dataset if `employee_attrition.csv` is not present. Running the project generates the EDA, model-comparison, ROC, precision-recall, confusion-matrix, permutation-importance and SHAP plots under `outputs/`.

## Run in Google Colab

1. Open `Employee_Attrition_Prediction.ipynb` in Google Colab.
2. Choose **Runtime → Run all**.
3. The dataset is loaded automatically from the public source if no local CSV is present.
4. Review the generated model comparison, threshold tuning, evaluation and explainability outputs.

## Run Locally

```bash
pip install -r requirements.txt
python run_project.py
```

Or open the notebook:

```bash
jupyter notebook Employee_Attrition_Prediction.ipynb
```

## Dataset

IBM HR Analytics Employee Attrition dataset: 1,470 rows and 35 original columns. Target: `Attrition` (`Yes` / `No`).

Public mirror used by the project: <https://github.com/nelson-wu/employee-attrition-ml>

## Responsible Use

This repository is an **academic/educational project**. Employee analytics is a high-impact domain. The model must not be used as the sole basis for real hiring, firing, promotion, compensation, disciplinary, or retention decisions.
