"""Train on OLD EDR cohort, freeze model, evaluate new challenge cohort without refitting."""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lab_common import (ROOT,load_data,split_indices,selected_classification_features,
                        prepare_features,classifier_metrics,out_name)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--algorithm',choices=['logistic','random_forest'],default='random_forest')
    ap.add_argument('--features',choices=['basic','extended'],default='basic')
    ap.add_argument('--threshold',type=float,default=.5)
    ap.add_argument('--name',default='old_model_shift')
    args=ap.parse_args()
    if not 0<=args.threshold<=1:ap.error('threshold must be 0..1')
    name=out_name(args.name)
    source,_=load_data('edr')
    shifted,_=load_data('edr',ROOT/'datasets'/'edr_events_challenge.csv')
    s=split_indices(source,'edr')
    features=selected_classification_features('edr',args.features)
    # Preprocessing and model are both fit ONLY on old-cohort training rows.
    full=pd.concat([source[features],shifted[features]],ignore_index=True)
    data,pre=prepare_features(full,features,use_scale=True,log_bytes=True)
    algorithm=(LogisticRegression(max_iter=1200,random_state=42) if args.algorithm=='logistic'
               else RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1))
    pipe=Pipeline([('preprocessing',pre),('model',algorithm)])
    pipe.fit(data.iloc[s['train']],source.iloc[s['train']]['malicious_label'])
    def calculate(df,x):
        scores=pipe.predict_proba(x)[:,1]
        pred=(scores>=args.threshold).astype(int)
        m=classifier_metrics(df['malicious_label'].to_numpy(),pred,scores)
        return m,df[['event_id','endpoint_type','process_name','malicious_label']].assign(
            predicted_label=pred,positive_probability=np.round(scores,6))
    old_metrics,old_preds=calculate(source.iloc[s['validation']],data.iloc[s['validation']])
    new_metrics,new_preds=calculate(shifted,data.iloc[len(source):])
    print('FROZEN OLD-COHORT MODEL: no fitting, preprocessing or threshold updates on new cohort.')
    print(f'ALGORITHM={args.algorithm}, FEATURES={args.features}, THRESHOLD={args.threshold}')
    print('COHORT                N       Precision   Recall      FPR      FP    FN    PR-AUC')
    for title,m in [('old_validation',old_metrics),('new_challenge',new_metrics)]:
        print(f'{title:20s} {m["n"]:5d}  {m["precision"]:9.3f}  {m["recall"]:7.3f}  {m["fpr"]:7.3f}  {m["fp"]:5d} {m["fn"]:5d} {m["pr_auc_average_precision"]:8.3f}')
    (ROOT/'outputs'/f'{name}_shift_metrics.json').write_text(json.dumps({
        'old_validation':old_metrics,'new_challenge':new_metrics,'features':features,
        'note':'New cohort synthetic; unknown field generalisation not claimed.'},indent=2))
    old_preds.to_csv(ROOT/'outputs'/f'{name}_old_validation.csv',index=False)
    new_preds.to_csv(ROOT/'outputs'/f'{name}_new_challenge.csv',index=False)
    labels=['old validation','new challenge']; vals=[[m[k] for m in (old_metrics,new_metrics)] for k in ['precision','recall','fpr']]
    fig,ax=plt.subplots(figsize=(7,4))
    idx=np.arange(2)
    for j,(metric,v) in enumerate(zip(['precision','recall','fpr'],vals)):
        ax.bar(idx+(j-1)*.23,v,width=.23,label=metric)
    ax.set_xticks(idx,labels);ax.set_ylim(0,1);ax.set_title('Frozen old model evaluated on a different synthetic cohort');ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'charts'/f'{name}_shift.png',dpi=130);plt.close(fig)
    print(f'Saved outputs/{name}_shift_metrics.json and charts/{name}_shift.png')
    print('CAUTION: differing population prevalence also affects precision; inspect FPR and recall separately.')

if __name__=='__main__':main()
