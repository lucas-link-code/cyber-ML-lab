"""Explicitly demonstrate POST-INVESTIGATION label leakage (never production features)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from lab_common import load_data,split_indices,prepare_features,selected_classification_features,classifier_metrics


def eval_model(frame,features,train,val):
    x,pre=prepare_features(frame,features)
    model=Pipeline([('preprocessor',pre),('model',LogisticRegression(max_iter=1200))])
    model.fit(x.iloc[train],frame.iloc[train]['phishing_label'])
    p=model.predict_proba(x.iloc[val])[:,1]
    return classifier_metrics(frame.iloc[val]['phishing_label'],(p>=.5).astype(int),p)


def main():
    frame,path=load_data('phishing','datasets/phishing_emails_leakage_challenge.csv')
    s=split_indices(frame,'phishing')
    safe=selected_classification_features('phishing','extended')
    legitimate=eval_model(frame,safe,s['train'],s['validation'])
    contaminated=eval_model(frame,safe+['post_investigation_decision'],s['train'],s['validation'])
    print('DELIBERATELY LEAKY CHALLENGE: post_investigation_decision is made AFTER investigation.')
    print('The feature is not available when an incoming email must be classified.')
    print('MODEL                   precision  recall    F1      PR-AUC')
    for name,metrics in [('valid_features_only',legitimate),('LEAKY_POST_OUTCOME',contaminated)]:
        print(f'{name:23s} {metrics["precision"]:8.3f} {metrics["recall"]:7.3f} {metrics["f1"]:7.3f} {metrics["pr_auc_average_precision"]:9.3f}')
    print('IMPORTANT: The apparently better leaky model is unusable for pre-investigation detection.')

if __name__=='__main__':main()
