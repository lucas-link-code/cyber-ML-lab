"""Shared experiment data preparation, deterministic splitting and reporting."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'phishing': 'phishing_emails.csv', 'edr': 'edr_events.csv',
            'triage': 'alert_triage.csv', 'network': 'network_activity.csv'}
LABELS = {'phishing': 'phishing_label', 'edr': 'malicious_label',
          'triage': 'estimated_triage_minutes', 'network': 'known_anomaly_label'}
IDS = {'phishing': 'email_id', 'edr': 'event_id', 'triage': 'alert_id', 'network': 'session_id'}
GROUPS = {'edr': 'hostname', 'network': 'source_host'}
RANDOM_SEED = 42


def load_data(dataset, dataset_file=None):
    path = Path(dataset_file) if dataset_file else ROOT / 'datasets' / DATASETS[dataset]
    if not path.is_file():
        raise FileNotFoundError(f'Dataset missing: {path}. Run: python scripts/generate_datasets.py')
    frame = pd.read_csv(path)
    if LABELS[dataset] not in frame:
        raise ValueError(f'Incorrect dataset file: required column {LABELS[dataset]!r} is missing')
    return frame, path


def split_indices(frame, dataset):
    """Stable disjoint 60/20/20 split; EDR/network groups stay exclusive."""
    ind = np.arange(len(frame))
    if dataset in GROUPS:
        groups = frame[GROUPS[dataset]].fillna('unknown').to_numpy()
        g1 = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=RANDOM_SEED)
        rest, test = next(g1.split(ind, groups=groups))
        g2 = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=RANDOM_SEED+1)
        tr_local, va_local = next(g2.split(rest, groups=groups[rest]))
        train, val = rest[tr_local], rest[va_local]
    else:
        y = frame[LABELS[dataset]] if dataset == 'phishing' else None
        rest, test = train_test_split(ind, test_size=.20, random_state=RANDOM_SEED, stratify=y)
        strat = frame.iloc[rest][LABELS[dataset]] if y is not None else None
        train, val = train_test_split(rest, test_size=.25, random_state=RANDOM_SEED+1, stratify=strat)
    assert len(set(train) & set(val)) == len(set(train) & set(test)) == len(set(val) & set(test)) == 0
    return {'train': np.asarray(train), 'validation': np.asarray(val), 'test': np.asarray(test)}


CLASSIFICATION_FEATURES = {
    'phishing': {
        'basic': ['sender_reputation', 'url_count', 'urgent_language_score', 'has_shortened_url', 'body_length'],
        'extended': ['sender_reputation', 'url_count', 'suspicious_url_count', 'urgent_language_score',
                     'has_shortened_url', 'body_length', 'attachment_count', 'subject_length',
                     'spf_result', 'dkim_result', 'dmarc_result', 'sender_domain'],
    },
    'edr': {
        'basic': ['command_line_length', 'process_prevalence', 'network_connection_count', 'unsigned_binary'],
        'extended': ['command_line_length', 'process_prevalence', 'network_connection_count',
                     'unique_destination_count', 'outbound_bytes', 'process_execution_frequency',
                     'unsigned_binary', 'unusual_parent_child', 'endpoint_type', 'process_name', 'parent_process'],
    }
}
REGRESSION_FEATURES = {
    'basic': ['telemetry_count', 'related_host_count', 'analyst_workload'],
    'extended': ['telemetry_count', 'related_host_count', 'analyst_workload',
                 'investigation_complexity', 'historical_alert_frequency', 'alert_type', 'alert_severity']
}
UNSUP_FEATURES = {
    'network': {
        'raw': ['connection_frequency', 'destination_count', 'bytes_sent', 'bytes_received',
                'connection_duration', 'unusual_hour_activity', 'failed_connection_count', 'destination_port'],
        'behavioral': ['connection_frequency', 'destination_count', 'bytes_sent',
                       'connection_duration', 'unusual_hour_activity', 'failed_connection_count'],
        'noisy': ['connection_frequency', 'destination_count', 'bytes_sent', 'bytes_received',
                  'connection_duration', 'unusual_hour_activity', 'failed_connection_count',
                  'destination_port', 'random_trace_tag'],
    },
    'edr': {
        'raw': ['command_line_length', 'process_prevalence', 'network_connection_count',
                'unique_destination_count', 'outbound_bytes', 'process_execution_frequency',
                'unsigned_binary', 'unusual_parent_child'],
        'behavioral': ['command_line_length', 'process_prevalence', 'network_connection_count',
                       'unique_destination_count', 'unsigned_binary', 'unusual_parent_child'],
        'noisy': ['command_line_length', 'process_prevalence', 'network_connection_count',
                  'unique_destination_count', 'outbound_bytes', 'process_execution_frequency',
                  'unsigned_binary', 'unusual_parent_child', 'random_trace_tag'],
    }
}


def selected_classification_features(dataset, mode):
    if mode in ('basic', 'extended'):
        return list(CLASSIFICATION_FEATURES[dataset][mode])
    if mode == 'text' and dataset == 'phishing':
        return CLASSIFICATION_FEATURES['phishing']['extended'] + ['subject_text', 'body_text']
    raise ValueError(f'Feature set {mode!r} not available for {dataset}')


def prepare_features(frame, features, use_scale=True, log_bytes=False):
    """Create learned imputers/encoders/scalers fitted on TRAIN ONLY."""
    prepared = frame[features].copy()
    texts = [f for f in features if f in ('subject_text', 'body_text')]
    for col in texts:
        prepared[col] = prepared[col].fillna('').astype(str)
    num = [f for f in features if pd.api.types.is_numeric_dtype(prepared[f])]
    cats = [f for f in features if f not in num and f not in texts]
    if log_bytes:
        for col in ('outbound_bytes', 'bytes_sent', 'bytes_received'):
            if col in prepared:
                prepared[col] = np.log1p(prepared[col].clip(lower=0))
    numeric_steps = [('impute', SimpleImputer(strategy='median'))]
    if use_scale:
        numeric_steps.append(('scale', StandardScaler()))
    transformers = []
    if num:
        transformers.append(('num', Pipeline(numeric_steps), num))
    if cats:
        transformers.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cats))
    for col in texts:
        transformers.append((f'text_{col}', TfidfVectorizer(max_features=180, min_df=3, ngram_range=(1,2)), col))
    preprocessing = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)
    return prepared, preprocessing


def classifier_metrics(y, pred, scores):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    result = {
        'n': int(len(y)), 'positives': int(np.sum(y)), 'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn), 'accuracy': float(accuracy_score(y,pred)),
        'precision': float(precision_score(y,pred,zero_division=0)),
        'recall': float(recall_score(y,pred,zero_division=0)),
        'f1': float(f1_score(y,pred,zero_division=0)),
        'fpr': float(fp/(fp+tn)) if (fp+tn)>0 else None,
        'predicted_positive_rate': float(np.mean(pred)),
    }
    result['roc_auc'] = float(roc_auc_score(y,scores)) if len(np.unique(y)) == 2 else None
    result['pr_auc_average_precision'] = float(average_precision_score(y,scores)) if len(np.unique(y))==2 else None
    return result


def regression_metrics(y, pred):
    mse = float(mean_squared_error(y,pred))
    return {'n': int(len(y)), 'mae': float(mean_absolute_error(y,pred)), 'mse': mse,
            'rmse': float(np.sqrt(mse)), 'r2': float(r2_score(y,pred))}


def out_name(raw):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', raw):
        raise ValueError('Experiment --name must be 1-64 characters: letters, digits, _ or -')
    return raw


def save_result(name, config, metrics, predictions, chart_fn=None):
    name = out_name(name)
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'charts').mkdir(exist_ok=True)
    payload = {'name': name, 'config': config, 'metrics': metrics}
    with (ROOT/'outputs'/f'{name}_metrics.json').open('w', encoding='utf8') as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
    predictions.to_csv(ROOT/'outputs'/f'{name}_predictions.csv', index=False)
    if chart_fn is not None:
        chart_fn(ROOT/'charts'/name)
    print('\nEXPERIMENT METRICS')
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f'  {key:27s} {value:.4f}')
        elif not isinstance(value, (dict, list)):
            print(f'  {key:27s} {value}')
    print(f'  Saved: outputs/{name}_metrics.json')
    print(f'  Saved: outputs/{name}_predictions.csv')
    if chart_fn is not None:
        print(f'  Charts: charts/{name}_*.png')


def format_table(title, table):
    print('\n' + title)
    print(table.to_string(index=False, max_colwidth=52))
