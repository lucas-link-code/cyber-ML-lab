"""Demonstrate GridSearchCV/RandomizedSearchCV on TRAIN ONLY, then evaluate validation."""
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, GroupKFold
from lab_common import (load_data,split_indices,selected_classification_features,prepare_features,
                        LABELS,GROUPS,IDS,classifier_metrics,save_result)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['phishing','edr'],default='phishing')
    ap.add_argument('--algorithm',choices=['logistic','random_forest'],default='logistic')
    ap.add_argument('--search',choices=['grid','random'],default='grid')
    ap.add_argument('--features',choices=['basic','extended'],default='extended')
    ap.add_argument('--name',default='tuned_classifier')
    args=ap.parse_args()
    frame,path=load_data(args.dataset)
    splits=split_indices(frame,args.dataset)
    tr=splits['train'];va=splits['validation'];y=frame[LABELS[args.dataset]]
    features=selected_classification_features(args.dataset,args.features)
    x,pre=prepare_features(frame,features,log_bytes=True)
    if args.algorithm=='logistic':
        alg=LogisticRegression(max_iter=1200,random_state=42)
        params={'model__C':[.05,.2,1,5], 'model__class_weight':[None,'balanced']}
    else:
        alg=RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=42)
        params={'model__max_depth':[4,8,14,None], 'model__min_samples_leaf':[1,4,12]}
    pipe=Pipeline([('preprocessing',pre),('model',alg)])
    if args.dataset in GROUPS:
        groups=frame.iloc[tr][GROUPS[args.dataset]]
        cv=GroupKFold(n_splits=3)
    else:
        groups=None; cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    opts={'estimator':pipe,'param_distributions':params,'n_iter':5} if args.search=='random' else {'estimator':pipe,'param_grid':params}
    if args.search=='random':
        search=RandomizedSearchCV(**opts,scoring='average_precision',cv=cv,n_jobs=-1,random_state=42,refit=True)
    else:
        search=GridSearchCV(**opts,scoring='average_precision',cv=cv,n_jobs=-1,refit=True)
    print(f'Fitting {args.search} search using TRAIN rows ONLY ({len(tr)})')
    print('Scoring: average_precision; 3-fold CV, group-exclusive folds for EDR.')
    search.fit(x.iloc[tr],y.iloc[tr],groups=groups)
    print('Best CV score:',round(search.best_score_,4),'Best parameters:',search.best_params_)
    prob=search.best_estimator_.predict_proba(x.iloc[va])[:,1]
    pred=(prob>=.5).astype(int)
    metrics=classifier_metrics(y.iloc[va],pred,prob)
    result=frame.iloc[va].copy()
    result.insert(1,'actual_label',y.iloc[va].to_numpy())
    result.insert(2,'predicted_label',pred)
    result.insert(3,'positive_probability',np.round(prob,6))
    result.insert(4,'error_type',np.select([(y.iloc[va].to_numpy()==0)&(pred==1),
                                             (y.iloc[va].to_numpy()==1)&(pred==0)],['FP','FN'],default='correct'))
    config={'problem':'classification','dataset':args.dataset,'dataset_file':path.name,
            'algorithm':args.algorithm,'features':args.features,'split':'validation','threshold':.5,
            'search':args.search,'best_params':search.best_params_,'best_cv_ap':float(search.best_score_)}
    save_result(args.name,config,metrics,result)

if __name__=='__main__':main()
