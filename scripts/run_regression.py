"""Predict continuous alert investigation time using supervised regression."""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lab_common import load_data,split_indices,REGRESSION_FEATURES,prepare_features,regression_metrics,save_result,format_table


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-file',default=None)
    ap.add_argument('--algorithm',choices=['linear','ridge','tree','random_forest','gradient_boosting'],default='linear')
    ap.add_argument('--features',choices=['basic','extended'],default='basic')
    ap.add_argument('--alpha',type=float,default=1.0)
    ap.add_argument('--max-depth',type=int,default=None)
    ap.add_argument('--n-estimators',type=int,default=120)
    ap.add_argument('--min-samples-leaf',type=int,default=3)
    ap.add_argument('--evaluate-test',action='store_true')
    ap.add_argument('--name',default='regression_baseline')
    args=ap.parse_args()
    if args.n_estimators<1 or args.min_samples_leaf<1 or args.alpha<=0:
        ap.error('Hyperparameters must be positive')
    frame,path=load_data('triage',args.dataset_file)
    target='estimated_triage_minutes'
    indices=split_indices(frame,'triage')
    train=indices['train']; scored=indices['test'] if args.evaluate_test else indices['validation']
    features=REGRESSION_FEATURES[args.features]
    data,pre=prepare_features(frame,features)
    model_map={
        'linear':LinearRegression(), 'ridge':Ridge(alpha=args.alpha),
        'tree':DecisionTreeRegressor(random_state=42,max_depth=args.max_depth,min_samples_leaf=args.min_samples_leaf),
        'random_forest':RandomForestRegressor(n_estimators=args.n_estimators,max_depth=args.max_depth,
                                     min_samples_leaf=args.min_samples_leaf,random_state=42,n_jobs=-1),
        'gradient_boosting':GradientBoostingRegressor(n_estimators=args.n_estimators,max_depth=args.max_depth or 3,random_state=42),
    }
    model=Pipeline([('preprocessing',pre),('model',model_map[args.algorithm])])
    model.fit(data.iloc[train],frame.iloc[train][target])
    actual=frame.iloc[scored][target].to_numpy()
    predicted=model.predict(data.iloc[scored])
    metrics=regression_metrics(actual,predicted)
    metrics['train_mae']=regression_metrics(frame.iloc[train][target],model.predict(data.iloc[train]))['mae']
    result=frame.iloc[scored].copy()
    result.insert(1,'actual_minutes',actual)
    result.insert(2,'predicted_minutes',np.round(predicted,2))
    result.insert(3,'residual_actual_minus_predicted',np.round(actual-predicted,2))
    print(f'ALGORITHM: {args.algorithm}; features={args.features}; split={"FINAL TEST" if args.evaluate_test else "validation"}')
    print('FEATURES:', ', '.join(features))
    format_table('SAMPLE PREDICTIONS',result[['alert_id','actual_minutes','predicted_minutes','residual_actual_minus_predicted']].head(12))
    config={'problem':'regression','dataset':'triage','dataset_file':path.name,'algorithm':args.algorithm,
            'features':args.features,'split':'test' if args.evaluate_test else 'validation',
            'alpha':args.alpha,'n_estimators':args.n_estimators,'max_depth':args.max_depth,
            'min_samples_leaf':args.min_samples_leaf}
    def plot(prefix):
        fig,ax=plt.subplots(figsize=(5,5));ax.scatter(actual,predicted,s=9,alpha=.35)
        lo=min(actual.min(),predicted.min());hi=max(actual.max(),predicted.max())
        ax.plot([lo,hi],[lo,hi],linestyle='--',color='black',lw=1);ax.set(xlabel='Actual minutes',ylabel='Predicted minutes',title='Prediction versus actual')
        fig.tight_layout();fig.savefig(f'{prefix}_pred_vs_actual.png',dpi=130);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4));ax.scatter(predicted,actual-predicted,s=9,alpha=.35)
        ax.axhline(0,color='black',ls='--');ax.set(xlabel='Predicted minutes',ylabel='Residual (actual - predicted)',title='Residual diagnostics')
        fig.tight_layout();fig.savefig(f'{prefix}_residuals.png',dpi=130);plt.close(fig)
    save_result(args.name,config,metrics,result,plot)

if __name__=='__main__':
    main()
