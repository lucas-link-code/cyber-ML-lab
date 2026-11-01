"""Anomaly scoring on held-out host groups; anomaly ground truth used ONLY to evaluate."""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from lab_common import (load_data,split_indices,UNSUP_FEATURES,LABELS,IDS,GROUPS,
                        prepare_features,classifier_metrics,save_result,format_table)


def make_estimator(name,contamination):
    if name=='isolation_forest':return IsolationForest(n_estimators=150,contamination=contamination,random_state=42,n_jobs=-1)
    if name=='lof':return LocalOutlierFactor(n_neighbors=30,contamination=contamination,novelty=True)
    if name=='one_class_svm':return OneClassSVM(gamma='scale',nu=contamination)
    if name=='zscore':return None
    raise ValueError(name)


def fit_scoring_model(xtrain,xscore,name,contamination):
    """A higher score always means MORE anomalous; threshold learned from train scores only."""
    if name=='zscore':
        mu=xtrain.mean(axis=0)
        sd=np.maximum(xtrain.std(axis=0),1e-9)
        train_scores=np.max(np.abs((xtrain-mu)/sd),axis=1)
        score=np.max(np.abs((xscore-mu)/sd),axis=1)
    else:
        detector=make_estimator(name,contamination)
        detector.fit(xtrain)
        train_scores=-detector.score_samples(xtrain)
        score=-detector.score_samples(xscore)
    threshold=np.quantile(train_scores,1-contamination)
    return score, threshold


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['network','edr'],default='network')
    ap.add_argument('--dataset-file',default=None)
    ap.add_argument('--algorithm',choices=['isolation_forest','lof','one_class_svm','zscore'],default='isolation_forest')
    ap.add_argument('--features',choices=['raw','behavioral','noisy'],default='raw')
    ap.add_argument('--baseline',choices=['global','per_role'],default='global')
    ap.add_argument('--contamination',type=float,default=.08)
    ap.add_argument('--scale',choices=['yes','no'],default='yes')
    ap.add_argument('--evaluate-test',action='store_true')
    ap.add_argument('--name',default='anomaly_baseline')
    args=ap.parse_args()
    if not 0<args.contamination<.5:ap.error('--contamination must be greater than 0 and less than 0.5')
    if args.algorithm=='zscore' and args.scale=='no':
        print('NOTE: Z-score standardizes features internally, regardless of --scale setting.')
    frame,path=load_data(args.dataset,args.dataset_file)
    indices=split_indices(frame,args.dataset)
    tr_idx=indices['train']; eval_idx=indices['test'] if args.evaluate_test else indices['validation']
    features=UNSUP_FEATURES[args.dataset][args.features]
    data,pre=prepare_features(frame,features,use_scale=args.scale=='yes',log_bytes=True)
    role_col='host_role' if args.dataset=='network' else 'endpoint_type'
    scores=np.empty(len(eval_idx),dtype=float)
    thresholds=np.empty(len(eval_idx),dtype=float)
    eval_roles=frame.iloc[eval_idx][role_col].to_numpy()
    train_roles=frame.iloc[tr_idx][role_col].to_numpy()
    print(f'TRAIN: {len(tr_idx)} | VALIDATION: {len(indices["validation"])} | TEST: {len(indices["test"])}')
    print('SPLIT: exclusive hosts. No anomaly labels provided to the detector.')
    print(f'Algorithm={args.algorithm}; features={args.features}; baseline={args.baseline}; contamination={args.contamination:.3f}')
    print('SCORING:', 'FINAL TEST (once)' if args.evaluate_test else 'VALIDATION')
    print('FEATURES:',', '.join(features))
    for role in (np.unique(eval_roles) if args.baseline=='per_role' else [None]):
        tr_rows=(train_roles==role) if role is not None else np.ones(len(tr_idx),dtype=bool)
        ev_rows=(eval_roles==role) if role is not None else np.ones(len(eval_idx),dtype=bool)
        if not np.any(ev_rows):continue
        if np.sum(tr_rows)<40: raise ValueError(f'Insufficient training rows in role: {role}')
        # A separate fit per role also learns separate imputations / scaling baselines.
        xtr=pre.fit_transform(data.iloc[tr_idx[tr_rows]])
        xev=pre.transform(data.iloc[eval_idx[ev_rows]])
        role_scores,role_thresh=fit_scoring_model(xtr,xev,args.algorithm,args.contamination)
        scores[ev_rows]=role_scores; thresholds[ev_rows]=role_thresh
        print(f'  baseline {role if role else "all roles"}: n_train={np.sum(tr_rows)} threshold={role_thresh:.4f}')
    pred=(scores>thresholds).astype(int)
    actual=frame.iloc[eval_idx][LABELS[args.dataset]].to_numpy()
    metrics=classifier_metrics(actual,pred,scores if args.baseline=='global' else (scores-thresholds))
    # Note: score magnitudes across separately fitted role models are incomparable;
    # threshold-relative scores are used only for a descriptive aggregate ranking/AUC.
    metrics['alert_count']=int(np.sum(pred))
    result=frame.iloc[eval_idx].copy()
    result.insert(1,'actual_label',actual)
    result.insert(2,'anomaly_prediction',pred)
    result.insert(3,'anomaly_score',np.round(scores,6))
    result.insert(4,'training_score_threshold',np.round(thresholds,6))
    result.insert(5,'score_minus_threshold',np.round(scores-thresholds,6))
    result.insert(6,'error_type',np.select([(actual==0)&(pred==1),(actual==1)&(pred==0)],['FP','FN'],default='correct'))
    print('\nALERT COUNTS BY ENDPOINT ROLE (ground truth only for evaluation):')
    tab=result.groupby(role_col).agg(total=('anomaly_prediction','size'),alerts=('anomaly_prediction','sum'),
                                    flagged_benign=('error_type',lambda x:(x=='FP').sum()))
    print(tab.to_string())
    format_table('TOP HIGH-SCORING SESSIONS / PROCESSES',result.sort_values('score_minus_threshold',ascending=False)
                  [[IDS[args.dataset],role_col,'actual_label','anomaly_prediction','score_minus_threshold','error_type']].head(12))
    config={'problem':'anomaly','dataset':args.dataset,'dataset_file':path.name,
            'algorithm':args.algorithm,'features':args.features,'baseline':args.baseline,
            'contamination':args.contamination,'scale':args.scale,
            'split':'test' if args.evaluate_test else 'validation'}
    def plot(prefix):
        fig,ax=plt.subplots(figsize=(7,4))
        ax.hist((scores-thresholds)[actual==0],bins=35,alpha=.55,label='Known normal',density=True)
        ax.hist((scores-thresholds)[actual==1],bins=35,alpha=.55,label='Known anomaly',density=True)
        ax.axvline(0,color='black',ls='--',label='Anomaly threshold')
        ax.set(xlabel='Anomaly score minus training threshold',ylabel='Density',title='Evaluation only: scores vs known labels')
        ax.legend();fig.tight_layout();fig.savefig(f'{prefix}_anomaly_scores.png',dpi=130);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        tab[['alerts','flagged_benign']].plot(kind='bar',ax=ax)
        ax.set(xlabel='Host role',ylabel='Event count',title='Alerts and false positives by population')
        fig.tight_layout();fig.savefig(f'{prefix}_role_alerts.png',dpi=130);plt.close(fig)
    save_result(args.name,config,metrics,result,plot)
    print('NOTE: "anomaly" is not equivalent to "malicious". Labels were evaluation-only.')

if __name__=='__main__':
    main()
