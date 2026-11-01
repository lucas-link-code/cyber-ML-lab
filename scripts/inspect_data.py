"""Level 1: inspect dataset rows, schema, missing values, class distribution."""
import argparse
import pandas as pd
from lab_common import load_data, LABELS, IDS, CLASSIFICATION_FEATURES, REGRESSION_FEATURES, UNSUP_FEATURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['phishing','edr','triage','network'], default='phishing')
    parser.add_argument('--dataset-file', default=None)
    parser.add_argument('--rows', type=int, default=10)
    args = parser.parse_args()
    frame, path = load_data(args.dataset,args.dataset_file)
    target = LABELS[args.dataset]
    print(f'DATASET: {path}\nSHAPE: {len(frame):,} rows x {len(frame.columns)} columns')
    print(f'RECORD IDENTIFIER: {IDS[args.dataset]} (never a feature)')
    print(f'TARGET: {target} (never a training input)')
    if args.dataset in CLASSIFICATION_FEATURES:
        print('BASELINE TRAINING FEATURES:', ', '.join(CLASSIFICATION_FEATURES[args.dataset]['basic']))
    elif args.dataset == 'triage':
        print('BASELINE TRAINING FEATURES:', ', '.join(REGRESSION_FEATURES['basic']))
    else:
        print('UNSUPERVISED INPUT FEATURES:', ', '.join(UNSUP_FEATURES['network']['raw']))
    print('\nFIRST ROWS (wide rows may wrap):')
    with pd.option_context('display.max_columns', None, 'display.width', 250, 'display.max_colwidth', 38):
        print(frame.head(max(1,args.rows)).to_string(index=False))
    print('\nCOLUMNS / DATA TYPES / MISSING:')
    for col in frame.columns:
        print(f' {col:29s} {str(frame[col].dtype):9s} {int(frame[col].isna().sum()):5d} missing')
    if args.dataset != 'triage':
        print('\nGROUND-TRUTH LABEL DISTRIBUTION (for evaluation; never unsupervised features):')
        print(frame[target].value_counts(dropna=False).sort_index().to_string())
        print(f'Positive class prevalence: {frame[target].mean():.2%}')
    else:
        print('\nTARGET SUMMARY (minutes):')
        print(frame[target].describe().round(2).to_string())


if __name__ == '__main__':
    main()
