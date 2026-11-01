"""Train a phishing or EDR classifier on train; score validation by default."""
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import export_text
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
from lab_common import (load_data, split_indices, LABELS, IDS, selected_classification_features,
                        prepare_features, classifier_metrics, save_result, format_table,
                        ROOT, out_name)


def make_model(args):
    weight = 'balanced' if args.class_weight == 'balanced' else None
    models = {
        'logistic': lambda: LogisticRegression(max_iter=1200, class_weight=weight, random_state=42, C=args.c),
        'tree': lambda: DecisionTreeClassifier(max_depth=args.max_depth, class_weight=weight,
                                               min_samples_leaf=args.min_samples_leaf, random_state=42),
        'random_forest': lambda: RandomForestClassifier(n_estimators=args.n_estimators,max_depth=args.max_depth,
                           min_samples_leaf=args.min_samples_leaf,class_weight=weight,random_state=42,n_jobs=-1),
        'naive_bayes': lambda: GaussianNB(),
        'knn': lambda: KNeighborsClassifier(n_neighbors=args.neighbors),
        'svm': lambda: SVC(kernel='rbf', probability=True, class_weight=weight, random_state=42),
        'gradient_boosting': lambda: GradientBoostingClassifier(n_estimators=args.n_estimators,
                                                                  max_depth=args.max_depth or 3,random_state=42),
    }
    if args.algorithm == 'xgboost':
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise SystemExit('Optional extension: install with "python -m pip install xgboost"') from exc
        return XGBClassifier(n_estimators=args.n_estimators,max_depth=args.max_depth or 4,
                             eval_metric='logloss',random_state=42,n_jobs=2)
    if args.algorithm in ['naive_bayes', 'knn', 'gradient_boosting', 'xgboost'] and weight:
        print('NOTE: --class-weight is unsupported/ignored by this algorithm.')
    return models[args.algorithm]()


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['phishing','edr'],default='phishing')
    ap.add_argument('--dataset-file',default=None)
    ap.add_argument('--algorithm',choices=['logistic','tree','random_forest','naive_bayes','knn','svm','gradient_boosting','xgboost'],default='logistic')
    ap.add_argument('--features',choices=['basic','extended','text'],default='basic')
    ap.add_argument('--threshold',type=float,default=.5)
    ap.add_argument('--class-weight',choices=['none','balanced'],default='none')
    ap.add_argument('--sampling',choices=['none','oversample'],default='none',help='Duplicate minority training rows only')
    ap.add_argument('--scale',choices=['yes','no'],default='yes')
    ap.add_argument('--n-estimators',type=int,default=100)
    ap.add_argument('--max-depth',type=int,default=None)
    ap.add_argument('--min-samples-leaf',type=int,default=1)
    ap.add_argument('--neighbors',type=int,default=5)
    ap.add_argument('--c',type=float,default=1.0)
    ap.add_argument('--evaluate-test',action='store_true',help='Use only after model selection (final holdout)')
    ap.add_argument('--name',default='classification_baseline')
    args=ap.parse_args()
    if not 0 <= args.threshold <= 1:
        ap.error('--threshold must be between 0 and 1')
    if args.features=='text' and args.dataset!='phishing':
        ap.error('Text features are only supported for phishing emails.')
    if args.n_estimators<1 or args.min_samples_leaf<1 or args.neighbors<1 or args.c<=0:
        ap.error('Hyperparameters must be positive.')
    return args


def _fmt(value):
    return format(float(value), '.10g')


def _clean(name):
    return str(name).replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')


def _feature_names(model):
    raw = model.named_steps['preprocessing'].get_feature_names_out()
    return [_clean(name) for name in raw]


def _scaling_line(model):
    numeric = model.named_steps['preprocessing'].named_transformers_.get('num')
    if isinstance(numeric, Pipeline) and 'scale' in numeric.named_steps:
        return 'numeric_scaling\tstandardized'
    return 'numeric_scaling\tnot applied'


def _feature_rows(names, columns, order=None):
    width = len(names)
    arrays = []
    for label, series in columns:
        arr = np.asarray(series, dtype=float).reshape(-1)
        if arr.shape != (width,):
            raise ValueError(f'{label} length {arr.shape[0]} does not match {width} features')
        arrays.append(arr)
    if order is None:
        order = np.arange(width)
    header = '\t'.join(['index', 'feature', *[label for label, _ in columns]])
    rows = [header]
    for i in order:
        vals = [_fmt(arr[i]) for arr in arrays]
        rows.append('\t'.join([str(int(i)), names[i], *vals]))
    return rows


def _linear_block(inner, names):
    coef = np.atleast_2d(np.asarray(inner.coef_, dtype=float))
    classes = list(getattr(inner, 'classes_', range(coef.shape[0])))
    lines = []
    intercept = np.atleast_1d(np.asarray(inner.intercept_, dtype=float))
    if intercept.shape[0] == 1:
        lines.append(f'intercept\t{_fmt(intercept[0])}')
    else:
        for i, value in enumerate(intercept):
            label = classes[i] if i < len(classes) else i
            lines.append(f'intercept_class_{label}\t{_fmt(value)}')
    if coef.shape[0] == 1 and len(classes) == 2:
        lines.append(f'positive_class\t{classes[1]}')
        lines.append('weight is the decision function coefficient for the positive class')
        lines.append('positive probability is the sigmoid of intercept + sum of preprocessor output times weight')
        columns = [('weight', coef[0])]
    elif coef.shape[0] == 1:
        columns = [('weight', coef[0])]
    else:
        columns = [(f'weight_class_{classes[i]}', coef[i]) for i in range(coef.shape[0])]
    order = np.argsort(-np.max(np.abs(coef), axis=0))
    lines.append('')
    lines.extend(_feature_rows(names, columns, order))
    return lines


def _nb_block(inner, names):
    lines = ['learned parameters are the class prior and the per feature mean and variance', '']
    variance = getattr(inner, 'var_', None)
    if variance is None:
        variance = inner.sigma_
    variance = np.asarray(variance, dtype=float)
    means = np.asarray(inner.theta_, dtype=float)
    for i, label in enumerate(inner.classes_):
        count = int(inner.class_count_[i])
        lines.append(f'class\t{label}\tprior\t{_fmt(inner.class_prior_[i])}\tcount\t{count}')
    columns = []
    for i, label in enumerate(inner.classes_):
        columns.append((f'mean_class_{label}', means[i]))
        columns.append((f'variance_class_{label}', variance[i]))
    lines.append('')
    lines.extend(_feature_rows(names, columns))
    return lines


def _knn_block(inner):
    return [
        'learned_weights\tnone',
        'KNeighborsClassifier stores the training rows and has no coefficient vector',
        f'n_neighbors\t{int(inner.n_neighbors)}',
        f'n_samples_fit\t{int(inner.n_samples_fit_)}',
        f'metric\t{inner.effective_metric_}',
    ]


def _svc_block(inner, names):
    lines = [
        'kernel\trbf',
        'dual_coefficient is the learned weight of that support vector',
        'remaining columns are that support vector in preprocessor output space',
        'fit_row indexes the matrix passed to fit',
    ]
    intercept = '\t'.join(_fmt(v) for v in np.atleast_1d(inner.intercept_))
    lines.append(f'intercept\t{intercept}')
    class_labels = np.repeat(inner.classes_, inner.n_support_)
    dual = np.atleast_2d(np.asarray(inner.dual_coef_, dtype=float))
    if dual.shape[0] == 1:
        dual_names = ['dual_coefficient']
        dual_rows = [dual[0]]
    else:
        dual_names = [f'dual_coefficient_{i}' for i in range(dual.shape[0])]
        dual_rows = list(dual)
    sv = np.asarray(inner.support_vectors_, dtype=float)
    fit_rows = np.asarray(inner.support_)
    lines.append('')
    lines.append('\t'.join(['support_index', 'fit_row', 'class_label', *dual_names, *names]))
    for i in range(sv.shape[0]):
        dual_vals = [_fmt(row[i]) for row in dual_rows]
        coords = [_fmt(v) for v in sv[i]]
        lines.append('\t'.join([
            str(i), str(int(fit_rows[i])), str(class_labels[i]), *dual_vals, *coords
        ]))
    return lines


def _append_tree(lines, title, estimator, names):
    lines.append(title)
    kwargs = {
        'feature_names': list(names),
        'max_depth': int(estimator.get_depth()),
        'decimals': 6,
        'show_weights': True,
    }
    if hasattr(estimator, 'classes_'):
        kwargs['class_names'] = [str(c) for c in estimator.classes_]
    lines.append(export_text(estimator, **kwargs).rstrip())
    lines.append('')


def _tree_block(inner, names, algorithm):
    lines = ['learned parameters are the split thresholds and the leaf values']
    if hasattr(inner, 'feature_importances_'):
        importance = np.asarray(inner.feature_importances_, dtype=float)
        lines.append('impurity_importance is the normalized impurity decrease, separate from the tree itself')
        lines.append('')
        lines.append('impurity_importance')
        lines.extend(_feature_rows(names, [('impurity_importance', importance)], np.argsort(-importance)))
        lines.append('')
    if algorithm == 'gradient_boosting':
        lines.append(f'learning_rate\t{_fmt(inner.learning_rate)}')
        lines.append('each stage leaf value is the raw tree output. The model multiplies it by learning_rate and adds the stages.')
        lines.append('')
        for i, estimator in enumerate(np.ravel(inner.estimators_)):
            _append_tree(lines, f'stage\t{i}', estimator, names)
    elif algorithm == 'random_forest':
        for i, estimator in enumerate(inner.estimators_):
            _append_tree(lines, f'tree\t{i}', estimator, names)
    else:
        _append_tree(lines, 'tree', inner, names)
    return lines


def _xgb_block(inner, names):
    lines = ['learned parameters are the boosted trees', '']
    importance = np.asarray(inner.feature_importances_, dtype=float)
    if importance.shape[0] == len(names):
        lines.append('gain_importance')
        lines.extend(_feature_rows(names, [('gain_importance', importance)], np.argsort(-importance)))
    else:
        fallback = [f'f{i}' for i in range(importance.shape[0])]
        lines.append('gain_importance')
        lines.extend(_feature_rows(fallback, [('gain_importance', importance)], np.argsort(-importance)))
    lines.append('')
    lines.append('feature_index')
    for i, name in enumerate(names):
        lines.append(f'f{i}\t{name}')
    lines.append('')
    for i, tree in enumerate(inner.get_booster().get_dump(with_stats=True)):
        lines.append(f'tree\t{i}')
        lines.append(tree.rstrip())
        lines.append('')
    return lines


def save_learned_weights(name, model, algorithm):
    name = out_name(name)
    inner = model.named_steps['model']
    names = _feature_names(model)
    lines = [
        f'algorithm\t{algorithm}',
        f'estimator\t{type(inner).__name__}',
        f'n_features\t{len(names)}',
        _scaling_line(model),
        'weights apply to preprocessor output',
    ]
    if hasattr(inner, 'classes_'):
        lines.append('classes\t' + '\t'.join(str(c) for c in inner.classes_))
    lines.append('')
    if algorithm == 'knn':
        lines.extend(_knn_block(inner))
    elif algorithm == 'svm':
        lines.extend(_svc_block(inner, names))
    elif algorithm == 'naive_bayes':
        lines.extend(_nb_block(inner, names))
    elif algorithm in ('tree', 'random_forest', 'gradient_boosting'):
        lines.extend(_tree_block(inner, names, algorithm))
    elif algorithm == 'xgboost':
        lines.extend(_xgb_block(inner, names))
    elif hasattr(inner, 'coef_'):
        lines.extend(_linear_block(inner, names))
    else:
        lines.append('this estimator has no coefficient vector')
    (ROOT / 'outputs').mkdir(exist_ok=True)
    path = ROOT / 'outputs' / f'{name}_weights.txt'
    path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf8')
    print(f'  Saved: outputs/{name}_weights.txt')


def main():
    args=parse_args()
    frame,path=load_data(args.dataset,args.dataset_file)
    split=split_indices(frame,args.dataset)
    label=LABELS[args.dataset]
    features=selected_classification_features(args.dataset,args.features)
    data,preprocessor=prepare_features(frame,features,use_scale=args.scale=='yes',log_bytes=True)
    train=split['train']; scored=split['test'] if args.evaluate_test else split['validation']
    model=Pipeline([('preprocessing',preprocessor),('model',make_model(args))])
    print(f'TRAIN: {len(train)} rows | VALIDATION: {len(split["validation"])} | FINAL TEST: {len(split["test"])}')
    print(f'SCORING: {"FINAL TEST (reserve for once-only confirmation)" if args.evaluate_test else "VALIDATION (use for tuning)"}')
    print('FEATURES:', ', '.join(features))
    print(f'ALGORITHM: {args.algorithm}; threshold={args.threshold}')
    fit_rows=train
    if args.sampling=='oversample':
        train_labels=frame.iloc[train][label].to_numpy()
        classes,counts=np.unique(train_labels,return_counts=True)
        if len(classes)!=2: raise ValueError('Oversampling requires both binary classes')
        minority=classes[np.argmin(counts)]
        minority_rows=train[train_labels==minority]
        extra=np.random.default_rng(42).choice(minority_rows,size=int(counts.max()-counts.min()),replace=True)
        fit_rows=np.concatenate([train,extra])
        print(f'OVERSAMPLING: duplicating {len(extra)} minority training rows; validation/test unchanged')
    model.fit(data.iloc[fit_rows],frame.iloc[fit_rows][label])
    scores=model.predict_proba(data.iloc[scored])[:,1]
    pred=(scores>=args.threshold).astype(int)
    actual=frame.iloc[scored][label].to_numpy()
    metrics=classifier_metrics(actual,pred,scores)
    tr_scores=model.predict_proba(data.iloc[train])[:,1]
    metrics['train_f1'] = float(classifier_metrics(frame.iloc[train][label],
                       (tr_scores>=args.threshold).astype(int),tr_scores)['f1'])
    predictions=frame.iloc[scored].copy()
    predictions.insert(1,'actual_label',actual)
    predictions.insert(2,'predicted_label',pred)
    predictions.insert(3,'positive_probability',np.round(scores,6))
    predictions.insert(4,'error_type',np.select([(actual==0)&(pred==1),(actual==1)&(pred==0)],
                                                ['FP','FN'],default='correct'))
    print('\nSAMPLE PREDICTIONS:')
    format_table('',predictions[[IDS[args.dataset],'actual_label','predicted_label','positive_probability','error_type']].head(12))
    print('\nCONFUSION MATRIX (rows=actual, columns=predicted)')
    print(pd.DataFrame([[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]],
                       index=['actual 0','actual 1'],columns=['pred 0','pred 1']).to_string())
    config={'problem':'classification','dataset':args.dataset,'dataset_file':str(path.name),
            'algorithm':args.algorithm,'features':args.features,'feature_columns':features,
            'threshold':args.threshold,'class_weight':args.class_weight,'sampling':args.sampling,
            'split':'test' if args.evaluate_test else 'validation','scale':args.scale,
            'n_estimators':args.n_estimators,'max_depth':args.max_depth,
            'min_samples_leaf':args.min_samples_leaf,'neighbors':args.neighbors,'c':args.c}
    def plot(prefix):
        fig,ax=plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay.from_predictions(actual,pred,labels=[0,1],ax=ax,cmap='Blues',colorbar=False)
        ax.set_title('Confusion matrix | '+args.name)
        fig.tight_layout();fig.savefig(f'{prefix}_confusion.png',dpi=130);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4))
        PrecisionRecallDisplay.from_predictions(actual,scores,ax=ax)
        ax.set_title('Precision-recall | '+args.name)
        fig.tight_layout();fig.savefig(f'{prefix}_precision_recall.png',dpi=130);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4))
        RocCurveDisplay.from_predictions(actual,scores,ax=ax)
        ax.set_title('ROC | '+args.name)
        fig.tight_layout();fig.savefig(f'{prefix}_roc.png',dpi=130);plt.close(fig)
        inner=model.named_steps['model']
        if hasattr(inner,'feature_importances_'):
            names=model.named_steps['preprocessing'].get_feature_names_out()
            order=np.argsort(inner.feature_importances_)[-14:]
            fig,ax=plt.subplots(figsize=(7,5))
            ax.barh(np.array(names)[order],inner.feature_importances_[order]);ax.set_xlabel('Impurity-based importance (not causal)')
            ax.set_title('Feature importances (training model)')
            fig.tight_layout();fig.savefig(f'{prefix}_feature_importance.png',dpi=130);plt.close(fig)
    save_result(args.name,config,metrics,predictions,plot)
    save_learned_weights(args.name, model, args.algorithm)

if __name__=='__main__':
    main()
