from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RS = 42
BASE = Path(__file__).resolve().parent
OUT, MODELS = BASE / 'outputs', BASE / 'models'
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
DATA_URL = 'https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv'
DATA_FILE = BASE / 'employee_attrition.csv'
df = pd.read_csv(DATA_FILE if DATA_FILE.exists() else DATA_URL)

print('Dataset shape:', df.shape)
print('Attrition rate: %.2f%%' % (100 * df['Attrition'].eq('Yes').mean()))

# ---------- EDA ----------
counts = df['Attrition'].value_counts().reindex(['No', 'Yes'])
plt.figure(figsize=(6,4)); plt.bar(counts.index, counts.values)
plt.title('Employee Attrition Distribution'); plt.ylabel('Employees'); plt.tight_layout()
plt.savefig(OUT/'01_attrition_distribution.png', dpi=150); plt.close()

ot = df.assign(Target=df['Attrition'].eq('Yes').astype(int)).groupby('OverTime')['Target'].mean()*100
plt.figure(figsize=(6,4)); plt.bar(ot.index, ot.values)
plt.title('Attrition Rate by Overtime'); plt.ylabel('Attrition Rate (%)'); plt.tight_layout()
plt.savefig(OUT/'02_attrition_by_overtime.png', dpi=150); plt.close()

# ---------- Preprocessing ----------
raw = df.copy()
drop_cols = ['EmployeeCount','EmployeeNumber','Over18','StandardHours','Gender']
data = df.drop(columns=drop_cols).copy()
data['Attrition'] = data['Attrition'].map({'No':0,'Yes':1})
X, y = data.drop(columns='Attrition'), data['Attrition']

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=.30, stratify=y, random_state=RS)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=.50, stratify=y_tmp, random_state=RS)

cat = X.select_dtypes(include='object').columns.tolist()
num = X.select_dtypes(exclude='object').columns.tolist()
pre = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat)
])

# ---------- 4-model comparison with 5-fold CV ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1500, class_weight='balanced', random_state=RS),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=RS),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced_subsample', random_state=RS, n_jobs=-1),
    'Extra Trees': ExtraTreesClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=RS, n_jobs=-1),
}
rows = []
for name, estimator in models.items():
    pipe = Pipeline([('preprocessor', pre), ('model', estimator)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
        scoring={'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc'}, n_jobs=-1)
    rows.append({'Model':name, **{k.replace('test_','CV '):v.mean() for k,v in scores.items() if k.startswith('test_')}})
cv_results = pd.DataFrame(rows).sort_values('CV f1', ascending=False)
cv_results.to_csv(OUT/'cv_results.csv', index=False)
print('\n5-Fold Cross Validation\n', cv_results.round(4).to_string(index=False))

cv_results.set_index('Model')[['CV accuracy','CV recall','CV f1','CV roc_auc']].plot(kind='bar', figsize=(10,5))
plt.ylim(0,1); plt.ylabel('Mean CV score'); plt.title('Model Comparison'); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(OUT/'03_model_comparison.png', dpi=150); plt.close()

# ---------- Hyperparameter tuning ----------
log_pipe = Pipeline([('preprocessor', pre), ('model', LogisticRegression(max_iter=1500, random_state=RS))])
grid = GridSearchCV(log_pipe, {
    'model__C':[.05,.2,1,5,20],
    'model__class_weight':[None,'balanced']
}, scoring='f1', cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
final_model = grid.best_estimator_
print('\nBest Logistic Regression params:', grid.best_params_)

# ---------- Validation-only threshold tuning ----------
val_prob = final_model.predict_proba(X_val)[:,1]
thresholds = np.arange(.20, .81, .01)
f1s = np.array([f1_score(y_val, val_prob >= t) for t in thresholds])
best_threshold = float(thresholds[f1s.argmax()])
print('Validation-selected threshold:', round(best_threshold, 2))

plt.figure(figsize=(7,4)); plt.plot(thresholds, f1s); plt.axvline(best_threshold, ls='--')
plt.xlabel('Decision threshold'); plt.ylabel('Validation F1'); plt.title('Threshold Optimization'); plt.tight_layout()
plt.savefig(OUT/'04_threshold_tuning.png', dpi=150); plt.close()

# ---------- Untouched test set ----------
test_prob = final_model.predict_proba(X_test)[:,1]
test_pred = (test_prob >= best_threshold).astype(int)
metrics = {
    'Accuracy':accuracy_score(y_test,test_pred),
    'Precision':precision_score(y_test,test_pred,zero_division=0),
    'Recall':recall_score(y_test,test_pred,zero_division=0),
    'F1':f1_score(y_test,test_pred,zero_division=0),
    'ROC-AUC':roc_auc_score(y_test,test_prob),
    'PR-AUC':average_precision_score(y_test,test_prob),
}
pd.DataFrame(metrics.items(), columns=['Metric','Score']).to_csv(OUT/'final_test_metrics.csv', index=False)
print('\nFinal Test Metrics\n', pd.Series(metrics).round(4).to_string())

ConfusionMatrixDisplay(confusion_matrix(y_test,test_pred), display_labels=['Stay','Leave']).plot(values_format='d')
plt.title('Final Confusion Matrix'); plt.tight_layout(); plt.savefig(OUT/'05_confusion_matrix.png', dpi=150); plt.close()

fpr,tpr,_ = roc_curve(y_test,test_prob)
plt.figure(figsize=(6,5)); plt.hline=None