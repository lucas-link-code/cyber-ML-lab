"""Check synthetic source-vs-challenge population shifts without using outcome labels."""
import argparse
import numpy as np
import pandas as pd
from lab_common import load_data, LABELS, IDS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['edr','triage','network'],default='edr')
    args=ap.parse_args()
    source,_=load_data(args.dataset)
    target,_=load_data(args.dataset, f'datasets/{"alert_triage" if args.dataset=="triage" else "network_activity" if args.dataset=="network" else "edr_events"}_challenge.csv')
    exclude={LABELS[args.dataset],IDS[args.dataset]}
    numeric=[c for c in source.select_dtypes(include=np.number).columns if c not in exclude]
    rows=[]
    for col in numeric:
        ref=source[col].dropna().to_numpy(); new=target[col].dropna().to_numpy()
        sigma=np.std(ref)
        delta=float((np.mean(new)-np.mean(ref))/sigma) if sigma>0 else 0
        rows.append((col,np.mean(ref),np.mean(new),delta))
    print('TRAINING POPULATION vs CHALLENGE POPULATION; sorted by absolute standardised mean shift')
    print('A large shift is a diagnostic clue, not proof of compromised behaviour.')
    print('No label column used in these diagnostics. Separate synthetic cohorts were generated.')
    print(pd.DataFrame(rows,columns=['feature','baseline_mean','challenge_mean','mean_shift_in_baseline_SD'])
          .sort_values('mean_shift_in_baseline_SD',key=abs,ascending=False).round(3).to_string(index=False))

if __name__=='__main__':main()
