"""Five-fold, training-only cross-validation for a fixed phishing/EDR classifier.

Examples:
  python scripts/run_cross_validation.py --dataset phishing --algorithm logistic --features extended --threshold 0.30 --name cv_logistic
  python scripts/run_cross_validation.py --dataset phishing --algorithm random_forest --features extended --threshold 0.30 --max-depth 12 --name cv_forest12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from lab_common import (
    GROUPS, LABELS, ROOT, classifier_metrics, load_data, out_name,
    prepare_features, selected_classification_features, split_indices,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['phishing', 'edr'], default='phishing')
    p.add_argument('--algorithm', choices=['logistic', 'random_forest'], default='logistic')
    p.add_argument('--features', choices=['basic', 'extended'], default='extended')
    p.add_argument('--threshold', type=float, default=0.30)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--max-depth', type=int, default=None)
    p.add_argument('--n-estimators', type=int, default=100)
    p.add_argument('--class-weight', choices=['none', 'balanced'], default='none')
    p.add_argument('--name', default='cross_validation')
    args = p.parse_args()
    if not 0 <= args.threshold <= 1:
        p.error('--threshold must be between 0 and 1')
    if args.folds < 2:
        p.error('--folds must be at least 2')
    if args.max_depth is not None and args.max_depth < 1:
        p.error('--max-depth must be positive')
    if args.n_estimators < 1:
        p.error('--n-estimators must be positive')
    out_name(args.name)
    return args


def make_estimator(args):
    weight = 'balanced' if args.class_weight == 'balanced' else None
    if args.algorithm == 'logistic':
        return LogisticRegression(C=1.0, class_weight=weight, max_iter=1200, random_state=42)
    return RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_samples_leaf=1, class_weight=weight, n_jobs=-1, random_state=42,
    )


def main():
    args = parse_args()
    frame, path = load_data(args.dataset)
    partitions = split_indices(frame, args.dataset)
    train_idx = partitions['train']
    cols = selected_classification_features(args.dataset, args.features)
    x, _ = prepare_features(frame, cols, log_bytes=True)
    y = frame[LABELS[args.dataset]]
    y_train = y.iloc[train_idx].to_numpy()
    if args.dataset in GROUPS:
        group_train = frame.iloc[train_idx][GROUPS[args.dataset]].fillna('unknown').to_numpy()
        if len(np.unique(group_train)) < args.folds:
            raise ValueError('Not enough distinct host groups for that number of folds')
        iterator = GroupKFold(n_splits=args.folds).split(x.iloc[train_idx], y_train, groups=group_train)
        cv_description = 'GroupKFold: hosts do not appear in both sides of a fold'
    else:
        iterator = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42).split(x.iloc[train_idx], y_train)
        cv_description = 'StratifiedKFold: class proportions are preserved across folds'

    print(f'DATASET: {path.name}; algorithm: {args.algorithm}; features: {args.features}; threshold: {args.threshold:.2f}')
    print(f'TRAINING POOL: {len(train_idx)} rows | reserved validation: {len(partitions["validation"])} | untouched final test: {len(partitions["test"])}')
    print(f'CROSS-VALIDATION: {args.folds} folds. {cv_description}')
    print('Each fold creates and fits a NEW model and preprocessing pipeline. Reserved validation and test rows are NOT used.')

    records = []
    for fold, (sub_train, sub_eval) in enumerate(iterator, start=1):
        fit_idx = train_idx[sub_train]
        eval_idx = train_idx[sub_eval]
        # New preprocessing object per fold: imputation, encoding, and scaling are fit on fold-training rows only.
        fold_x, fold_preprocessor = prepare_features(frame, cols, log_bytes=True)
        pipe = Pipeline([('preprocessing', fold_preprocessor), ('model', make_estimator(args))])
        pipe.fit(fold_x.iloc[fit_idx], y.iloc[fit_idx])
        p_eval = pipe.predict_proba(fold_x.iloc[eval_idx])[:, 1]
        m = classifier_metrics(y.iloc[eval_idx], (p_eval >= args.threshold).astype(int), p_eval)
        p_train = pipe.predict_proba(fold_x.iloc[fit_idx])[:, 1]
        m_train = classifier_metrics(y.iloc[fit_idx], (p_train >= args.threshold).astype(int), p_train)
        records.append({'fold': fold, 'train_rows': len(fit_idx), 'fold_eval_rows': len(eval_idx),
                        'train_f1': m_train['f1'], 'f1': m['f1'],
                        'precision': m['precision'], 'recall': m['recall'],
                        'accuracy': m['accuracy'], 'tp': m['tp'], 'fp': m['fp'],
                        'fn': m['fn'], 'tn': m['tn'], 'pr_auc_average_precision': m['pr_auc_average_precision']})
        print(f'Fold {fold}: train F1={m_train["f1"]:.3f}, held-out F1={m["f1"]:.3f}, '
              f'precision={m["precision"]:.3f}, recall={m["recall"]:.3f}, '
              f'TP={m["tp"]}, FP={m["fp"]}, FN={m["fn"]}')

    folds = pd.DataFrame(records)
    metrics = ['train_f1', 'f1', 'precision', 'recall', 'accuracy', 'pr_auc_average_precision']
    summary = {'config': {'dataset': args.dataset, 'algorithm': args.algorithm,
                          'features': args.features, 'threshold': args.threshold,
                          'folds': args.folds, 'max_depth': args.max_depth,
                          'n_estimators': args.n_estimators, 'class_weight': args.class_weight,
                          'random_seed': 42, 'source_split': 'training_only'},
               'mean': {m: float(folds[m].mean()) for m in metrics},
               'sample_std': {m: float(folds[m].std(ddof=1)) for m in metrics},
               'min': {m: float(folds[m].min()) for m in metrics},
               'max': {m: float(folds[m].max()) for m in metrics}}
    print('\nSUMMARY (held-out folds within the training pool):')
    for m in metrics:
        print(f'  {m:28s} mean={summary["mean"][m]:.4f}  sample_std={summary["sample_std"][m]:.4f} '
              f' range=[{summary["min"][m]:.4f}, {summary["max"][m]:.4f}]')
    print('\nThese scores estimate variability across partitions, not independent deployment environments or a confidence interval.')

    out = ROOT / 'outputs'; charts = ROOT / 'charts'
    out.mkdir(exist_ok=True); charts.mkdir(exist_ok=True)
    folds.to_csv(out / f'{args.name}_cv_folds.csv', index=False)
    (out / f'{args.name}_cv_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(1, len(folds)+1)
    ax.plot(positions, folds['train_f1'], marker='o', label='Training F1')
    ax.plot(positions, folds['f1'], marker='s', label='Held-out fold F1')
    ax.set(xlabel='Fold', ylabel='F1 score', title=f'{args.algorithm}: cross-validation F1 by fold')
    ax.set_xticks(positions); ax.set_ylim(0, 1.05); ax.legend(); fig.tight_layout()
    fig.savefig(charts / f'{args.name}_cv_f1.png', dpi=130)
    plt.close(fig)
    print(f'\nSaved: outputs/{args.name}_cv_folds.csv, outputs/{args.name}_cv_summary.json, charts/{args.name}_cv_f1.png')


if __name__ == '__main__':
    main()
