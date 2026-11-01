"""Inspect actual false positives, false negatives or largest regression residuals."""
import argparse
import json
import pandas as pd
from lab_common import ROOT,out_name


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',required=True)
    ap.add_argument('--type',choices=['fp','fn','all_errors','largest_residuals','top_alerts'],default='fp')
    ap.add_argument('--rows',type=int,default=12)
    args=ap.parse_args();name=out_name(args.experiment)
    file=ROOT/'outputs'/f'{name}_predictions.csv'
    meta=ROOT/'outputs'/f'{name}_metrics.json'
    if not file.exists() or not meta.exists():ap.error('Run the named experiment first')
    cfg=json.loads(meta.read_text())['config']
    df=pd.read_csv(file)
    if args.type in ('fp','fn','all_errors'):
        if 'error_type' not in df:ap.error('This experiment does not contain FP/FN classification errors')
        match={'fp':['FP'],'fn':['FN'],'all_errors':['FP','FN']}[args.type]
        df=df[df['error_type'].isin(match)]
    elif args.type=='largest_residuals':
        if 'residual_actual_minus_predicted' not in df:ap.error('Requires regression experiment')
        df=df.reindex(df['residual_actual_minus_predicted'].abs().sort_values(ascending=False).index)
    else:
        if 'score_minus_threshold' not in df:ap.error('Requires anomaly experiment')
        df=df.sort_values('score_minus_threshold',ascending=False)
    print(f'EXPERIMENT: {name} | {cfg["problem"]} | {cfg["split"]} | match count: {len(df)}')
    cols=[c for c in df.columns if c not in ('body_text','subject_text')]
    with pd.option_context('display.max_columns',None,'display.width',230,'display.max_colwidth',35):
        print(df[cols].head(max(1,args.rows)).to_string(index=False))

if __name__=='__main__':main()
