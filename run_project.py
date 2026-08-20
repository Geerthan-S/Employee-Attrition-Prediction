from pathlib import Path
import warnings; warnings.filterwarnings('ignore')
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd, shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RS=42; BASE=Path(__file__).resolve().parent; OUT=BASE/'outputs'; MOD=BASE/'models'; OUT.mkdir(exist_ok=True); MOD.mkdir(exist_ok=True)
URL='https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv'
local=BASE/'employee_attrition.csv'; df=pd.read_csv(local if local.exists() else URL)
print('Dataset:',df.shape,' Attrition rate:',round(100*df.Attrition.eq('Yes').mean(),2),'%')

# EDA
c=df.Attrition.value_counts().reindex(['No','Yes']); plt.bar(c.index,c.values); plt.title('Employee Attrition Distribution'); plt.tight_layout(); plt.savefig(OUT/'01_attrition_distribution.png'); plt.close()
ot=df.assign(y=df.Attrition.eq('Yes')).groupby('OverTime').y.mean()*100; plt.bar(ot.index,ot.values); plt.title('Attrition Rate by Overtime'); plt.ylabel('%'); plt.tight_layout(); plt.savefig(OUT/'02_attrition_by_overtime.png'); plt.close()

# Prepare data; Gender is excluded from predictors and used only for post-hoc diagnostic
raw=df.copy(); drop=['EmployeeCount','EmployeeNumber','Over18','StandardHours','Gender']; d=df.drop(columns=drop).copy(); d['Attrition']=d.Attrition.map({'No':0,'Yes':1}); X=d.drop(columns='Attrition'); y=d.Attrition
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.30,stratify=y,random_state=RS); Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.50,stratify=ytmp,random_state=RS)
cat=X.select_dtypes('object').columns.tolist(); num=X.select_dtypes(exclude='object').columns.tolist()
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
cv=StratifiedKFold(5,shuffle=True,random_state=RS)
models={'Logistic Regression':LogisticRegression(max_iter=1500,class_weight='balanced',random_state=RS),'Decision Tree':DecisionTreeClassifier(max_depth=5,class_weight='balanced',random_state=RS),'Random Forest':RandomForestClassifier(n_estimators=120,max_depth=8,class_weight='balanced_subsample',random_state=RS,n_jobs=-1),'Extra Trees':ExtraTreesClassifier(n_estimators=120,max_depth=10,class_weight='balanced',random_state=RS,n_jobs=-1)}
rows=[]
for n,m in models.items():
 p=Pipeline([('pre',pre),('model',m)]); s=cross_validate(p,Xtr,ytr,cv=cv,scoring={'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc'},n_jobs=-1); rows.append({'Model':n,**{k[5:]:v.mean() for k,v in s.items() if k.startswith('test_')}})
cvdf=pd.DataFrame(rows).sort_values('f1',ascending=False); cvdf.to_csv(OUT/'cv_results.csv',index=False); print('\nCV results:\n',cvdf.round(4).to_string(index=False)); cvdf.set_index('Model')[['accuracy','recall','f1','roc_auc']].plot.bar(figsize=(10,5)); plt.ylim(0,1); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT/'03_model_comparison.png'); plt.close()

# Hyperparameter tuning + validation-only threshold selection
pipe=Pipeline([('pre',pre),('model',LogisticRegression(max_iter=1500,random_state=RS))]); grid=GridSearchCV(pipe,{'model__C':[.05,.2,1,5,20],'model__class_weight':[None,'balanced']},scoring='f1',cv=cv,n_jobs=-1); grid.fit(Xtr,ytr); model=grid.best_estimator_; print('\nBest params:',grid.best_params_)
vp=model.predict_proba(Xv)[:,1]; ts=np.arange(.20,.81,.01); fs=np.array([f1_score(yv,vp>=t) for t in ts]); th=float(ts[fs.argmax()]); print('Best validation threshold:',round(th,2)); plt.plot(ts,fs); plt.axvline(th,ls='--'); plt.xlabel('Threshold'); plt.ylabel('Validation F1'); plt.tight_layout(); plt.savefig(OUT/'04_threshold_tuning.png'); plt.close()

# Untouched test evaluation
prob=model.predict_proba(Xte)[:,1]; pred=(prob>=th).astype(int); metrics={'Accuracy':accuracy_score(yte,pred),'Precision':precision_score(yte,pred,zero_division=0),'Recall':recall_score(yte,pred,zero_division=0),'F1':f1_score(yte,pred,zero_division=0),'ROC-AUC':roc_auc_score(yte,prob),'PR-AUC':average_precision_score(yte,prob)}; pd.DataFrame(metrics.items(),columns=['Metric','Score']).to_csv(OUT/'final_test_metrics.csv',index=False); print('\nTest metrics:\n',pd.Series(metrics).round(4).to_string())
ConfusionMatrixDisplay(confusion_matrix(yte,pred),display_labels=['Stay','Leave']).plot(); plt.tight_layout(); plt.savefig(OUT/'05_confusion_matrix.png'); plt.close()
fpr,tpr,_=roc_curve(yte,prob); plt.plot(fpr,tpr,label=f"AUC={metrics['ROC-AUC']:.3f}"); plt.plot([0,1],[0,1],ls='--'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'06_roc_curve.png'); plt.close()
pr,rc,_=precision_recall_curve(yte,prob); plt.plot(rc,pr,label=f"PR-AUC={metrics['PR-AUC']:.3f}"); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'07_precision_recall_curve.png'); plt.close()

# Explainability
perm=permutation_importance(model,Xte,yte,scoring='f1',n_repeats=5,random_state=RS,n_jobs=-1); imp=pd.DataFrame({'Feature':X.columns,'Importance':perm.importances_mean}).sort_values('Importance',ascending=False); imp.to_csv(OUT/'permutation_importance.csv',index=False); top=imp.head(12).sort_values('Importance'); plt.barh(top.Feature,top.Importance); plt.tight_layout(); plt.savefig(OUT/'08_permutation_importance.png'); plt.close()
prep=model.named_steps['pre']; est=model.named_steps['model']; names=prep.get_feature_names_out(); bg=prep.transform(Xtr.sample(100,random_state=RS)); ex=prep.transform(Xte.iloc[:60]); sv=shap.Explainer(est,bg,feature_names=names)(ex); sv=sv[:,:,1] if sv.values.ndim==3 else sv; shap.plots.bar(sv,max_display=12,show=False); plt.tight_layout(); plt.savefig(OUT/'09_shap_global_importance.png'); plt.close()

# Small subgroup diagnostic
ge=raw.loc[Xte.index,'Gender']; fair=[]
for g in sorted(ge.unique()):
 m=ge.values==g; fair.append({'Group':g,'Samples':int(m.sum()),'Accuracy':accuracy_score(yte.values[m],pred[m]),'Recall':recall_score(yte.values[m],pred[m],zero_division=0),'F1':f1_score(yte.values[m],pred[m],zero_division=0)})
pd.DataFrame(fair).to_csv(OUT/'fairness_diagnostic.csv',index=False)
joblib.dump({'pipeline':model,'decision_threshold':th,'feature_columns':X.columns.tolist(),'excluded_columns':drop},MOD/'attrition_model.joblib')
print('\nTop features:\n',imp.head(8).round(4).to_string(index=False)); print('\nDone. Outputs:',OUT)
