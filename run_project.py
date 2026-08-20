import os, warnings, shutil
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND']='Agg'
import pandas as pd, numpy as np, matplotlib.pyplot as plt, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance
import shap

BASE=Path(__file__).resolve().parent
OUT=BASE/'outputs'; MOD=BASE/'models'; OUT.mkdir(exist_ok=True); MOD.mkdir(exist_ok=True)
DATA_URL='https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv'
DATA_FILE=BASE/'employee_attrition.csv'
df=pd.read_csv(DATA_FILE if DATA_FILE.exists() else DATA_URL)
RS=42
# EDA figs
counts=df['Attrition'].value_counts().reindex(['No','Yes'])
plt.figure(figsize=(6,4)); plt.bar(counts.index,counts.values); plt.title('Employee Attrition Distribution'); plt.xlabel('Attrition'); plt.ylabel('Employees'); plt.tight_layout(); plt.savefig(OUT/'01_attrition_distribution.png',dpi=160); plt.close()
rate_ot=df.assign(AttritionFlag=df['Attrition'].eq('Yes').astype(int)).groupby('OverTime')['AttritionFlag'].mean().mul(100).sort_values(ascending=False)
plt.figure(figsize=(6,4)); plt.bar(rate_ot.index,rate_ot.values); plt.title('Attrition Rate by Overtime'); plt.xlabel('OverTime'); plt.ylabel('Attrition Rate (%)'); plt.tight_layout(); plt.savefig(OUT/'02_attrition_by_overtime.png',dpi=160); plt.close()
role_rate=df.assign(AttritionFlag=df['Attrition'].eq('Yes').astype(int)).groupby('JobRole')['AttritionFlag'].mean().mul(100).sort_values()
plt.figure(figsize=(9,5)); plt.barh(role_rate.index,role_rate.values); plt.title('Attrition Rate by Job Role'); plt.xlabel('Attrition Rate (%)'); plt.tight_layout(); plt.savefig(OUT/'03_attrition_by_job_role.png',dpi=160); plt.close()
plt.figure(figsize=(8,5)); plt.hist(df.loc[df.Attrition=='No','MonthlyIncome'],bins=25,alpha=.55,label='Stay'); plt.hist(df.loc[df.Attrition=='Yes','MonthlyIncome'],bins=25,alpha=.55,label='Leave'); plt.title('Monthly Income Distribution by Attrition'); plt.xlabel('Monthly Income'); plt.ylabel('Frequency'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'04_income_distribution.png',dpi=160); plt.close()

raw=df.copy(); drop=['EmployeeCount','EmployeeNumber','Over18','StandardHours','Gender']; data=df.drop(columns=drop).copy(); data['Attrition']=data['Attrition'].map({'Yes':1,'No':0}); X=data.drop(columns='Attrition'); y=data['Attrition']
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.30,random_state=RS,stratify=y); Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.50,random_state=RS,stratify=ytmp)
cat=X.select_dtypes(include='object').columns.tolist(); num=X.select_dtypes(exclude='object').columns.tolist()
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
cv=StratifiedKFold(5,shuffle=True,random_state=RS)
models={'Logistic Regression':LogisticRegression(max_iter=1500,class_weight='balanced',random_state=RS),'Decision Tree':DecisionTreeClassifier(max_depth=5,class_weight='balanced',random_state=RS),'Random Forest':RandomForestClassifier(n_estimators=100,max_depth=8,class_weight='balanced_subsample',random_state=RS,n_jobs=1),'Extra Trees':ExtraTreesClassifier(n_estimators=100,max_depth=10,class_weight='balanced',random_state=RS,n_jobs=1)}
rows=[]
for name,m in models.items():
    pipe=Pipeline([('preprocessor',pre),('model',m)])
    sc=cross_validate(pipe,Xtr,ytr,cv=cv,scoring={'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc'},n_jobs=1)
    rows.append({'Model':name,'CV Accuracy':sc['test_accuracy'].mean(),'CV Precision':sc['test_precision'].mean(),'CV Recall':sc['test_recall'].mean(),'CV F1':sc['test_f1'].mean(),'CV ROC-AUC':sc['test_roc_auc'].mean()})
cvdf=pd.DataFrame(rows).sort_values('CV F1',ascending=False); cvdf.to_csv(OUT/'cv_results.csv',index=False)
ax=cvdf.set_index('Model')[['CV Accuracy','CV Recall','CV F1','CV ROC-AUC']].plot(kind='bar',figsize=(11,5)); plt.title('5-Fold Cross-Validation Model Comparison'); plt.ylabel('Mean CV Score'); plt.ylim(0,1); plt.xticks(rotation=0); plt.legend(loc='lower right'); plt.close()