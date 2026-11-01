"""Validate dataset schema, non-leakage and host-exclusive splits."""
import hashlib
import tempfile
from pathlib import Path
import numpy as np
from lab_common import (ROOT,load_data,split_indices,DATASETS,LABELS,IDS,GROUPS,
                        CLASSIFICATION_FEATURES,REGRESSION_FEATURES,UNSUP_FEATURES)


def main():
    expected={'phishing':6000,'edr':6500,'triage':3500,'network':5000}
    for dataset,filename in DATASETS.items():
        df,path=load_data(dataset)
        assert len(df)==expected[dataset], (dataset,len(df))
        assert df[IDS[dataset]].is_unique
        assert df[LABELS[dataset]].notna().all()
        assert df.isna().sum().sum()>0, 'Missingness intentionally present'
        splits=split_indices(df,dataset)
        names=list(splits)
        assert sum(len(splits[k]) for k in names)==len(df)
        for i,k in enumerate(names):
            for other in names[i+1:]:
                assert set(splits[k]).isdisjoint(splits[other]),(dataset,k,other)
                if dataset in GROUPS:
                    g=GROUPS[dataset]
                    assert set(df.iloc[splits[k]][g]).isdisjoint(df.iloc[splits[other]][g]),(dataset,g,k,other)
        feature_sets=(CLASSIFICATION_FEATURES[dataset].values() if dataset in CLASSIFICATION_FEATURES
                      else REGRESSION_FEATURES.values() if dataset=='triage' else UNSUP_FEATURES[dataset].values())
        for fset in feature_sets:
            assert LABELS[dataset] not in fset and IDS[dataset] not in fset
            assert all(name in df.columns for name in fset)
        if dataset=='phishing': assert df['sender_domain'].str.endswith('.example').all()
        if dataset!='triage':assert set(df[LABELS[dataset]].unique())=={0,1}
        size=path.stat().st_size
        print(f'PASS {filename:26s} rows={len(df):5d} bytes={size:8d}; disjoint splits and clean feature lists')
    print('PASS synthetic dataset structural checks. No final-test model training is part of this script.')

if __name__=='__main__':main()
