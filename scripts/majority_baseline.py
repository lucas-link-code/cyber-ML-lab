"""Show how high accuracy can hide zero malicious-class recall."""
import argparse
import numpy as np
from lab_common import load_data, split_indices, LABELS, classifier_metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['phishing','edr'],default='phishing')
    args=ap.parse_args();df,_=load_data(args.dataset)
    ix=split_indices(df,args.dataset)['validation']
    y=df.iloc[ix][LABELS[args.dataset]].to_numpy()
    pred=np.zeros_like(y)
    print('PREDICT-EVERYTHING-BENIGN BASELINE: no classifier trained')
    print('This deliberately useless baseline illustrates class imbalance.')
    m=classifier_metrics(y,pred,np.zeros(len(y)))
    for k in ['accuracy','precision','recall','f1','tp','fp','fn','tn']:
        print(f'{k:15}: {m[k]:.4f}' if isinstance(m[k],float) else f'{k:15}: {m[k]}')

if __name__=='__main__':main()
