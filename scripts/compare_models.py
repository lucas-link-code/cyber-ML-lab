"""Compare previously saved experiments from the same dataset and evaluation split."""
import argparse
import json
import pandas as pd
from lab_common import ROOT, out_name


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('experiments',nargs='+',help='Experiment --name values; specify at least two')
    ap.add_argument('--name',default='comparison')
    args=ap.parse_args()
    if len(args.experiments)<2:ap.error('Supply at least 2 experiment names')
    rows=[];base=None
    for name in args.experiments:
        name=out_name(name)
        file=ROOT/'outputs'/f'{name}_metrics.json'
        if not file.exists(): ap.error(f'Missing {file.name}: run the experiment first')
        record=json.loads(file.read_text(encoding='utf8'))
        cfg=record['config']
        signature=(cfg.get('problem'),cfg.get('dataset'),cfg.get('dataset_file'),cfg.get('split'))
        if base is None:base=signature
        if signature!=base:ap.error(f'Cannot compare different problem/dataset/split: {name} {signature} != {base}')
        rows.append({'experiment':name,'algorithm':cfg.get('algorithm'),
                     'features':cfg.get('features'),'threshold':cfg.get('threshold'),
                     'class_weight':cfg.get('class_weight'),'baseline':cfg.get('baseline'),
                     'contamination':cfg.get('contamination'),**record['metrics']})
    df=pd.DataFrame(rows)
    metrics={'classification':['accuracy','precision','recall','f1','fpr','fp','fn','tp','tn','roc_auc','pr_auc_average_precision','train_f1'],
             'regression':['mae','rmse','mse','r2','train_mae'],
             'anomaly':['precision','recall','f1','fpr','fp','fn','alert_count'],
             'clustering':['n_clusters','noise_fraction','silhouette_sample','inertia']}
    columns=[c for c in ['experiment','algorithm','features','threshold','class_weight','baseline','contamination']+metrics[base[0]] if c in df and not df[c].isna().all()]
    print('COMPARISON (same dataset and split):')
    print(df[columns].round(4).to_string(index=False))
    target=ROOT/'outputs'/f'{out_name(args.name)}.csv'
    df[columns].to_csv(target,index=False)
    print(f'\nSaved: outputs/{target.name}')
    print('Interpretation: changing threshold trades off FP versus FN; changing features or fit may change ranking quality.')
    print('Do not choose parameters using final-test performance; use validation or cross-validation.')

if __name__=='__main__':
    main()
