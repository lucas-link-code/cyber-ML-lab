"""Explore unlabeled behavioural clusters; ground-truth labels are NEVER features."""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from lab_common import (load_data, UNSUP_FEATURES, IDS, LABELS, prepare_features,
                        save_result,format_table)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['network','edr'],default='network')
    ap.add_argument('--dataset-file',default=None)
    ap.add_argument('--algorithm',choices=['kmeans','dbscan','hierarchical'],default='kmeans')
    ap.add_argument('--features',choices=['raw','behavioral','noisy'],default='behavioral')
    ap.add_argument('--scale',choices=['yes','no'],default='yes')
    ap.add_argument('--k',type=int,default=4)
    ap.add_argument('--eps',type=float,default=1.6)
    ap.add_argument('--min-samples',type=int,default=12)
    ap.add_argument('--elbow',action='store_true',help='Plot k=2 through 9 inertias (K-means only)')
    ap.add_argument('--name',default='clustering_baseline')
    args=ap.parse_args()
    if args.k<2 or args.eps<=0 or args.min_samples<1: ap.error('k>=2, eps>0, min-samples>=1')
    frame,path=load_data(args.dataset,args.dataset_file)
    features=UNSUP_FEATURES[args.dataset][args.features]
    data,pre=prepare_features(frame,features,use_scale=args.scale=='yes',log_bytes=True)
    x=pre.fit_transform(data)
    print(f'UNSUPERVISED: {len(frame)} events | {x.shape[1]} transformed dimensions')
    print('TRAINING FEATURES (NO LABELS):',', '.join(features))
    print(f'ALGORITHM: {args.algorithm}, scaling={args.scale}')
    if args.algorithm=='kmeans':
        model=KMeans(n_clusters=args.k,n_init=10,random_state=42)
        cluster=model.fit_predict(x)
    elif args.algorithm=='dbscan':
        model=DBSCAN(eps=args.eps,min_samples=args.min_samples)
        cluster=model.fit_predict(x)
    else:
        model=AgglomerativeClustering(n_clusters=args.k)
        cluster=model.fit_predict(x)
    unique=np.unique(cluster)
    nonnoise=cluster>=0
    distinct=np.unique(cluster[nonnoise])
    valid=(len(distinct)>=2 and np.sum(nonnoise)>=30 and np.sum(nonnoise)<10001)
    if valid:
        take=np.random.default_rng(42).choice(np.where(nonnoise)[0],size=min(np.sum(nonnoise),850),replace=False)
        try: silhouette=float(silhouette_score(x[take],cluster[take]))
        except ValueError: silhouette=None
    else: silhouette=None
    metrics={'n':int(len(frame)),'n_clusters':int(len(distinct)),
             'noise_count':int(np.sum(~nonnoise)),'noise_fraction':float(np.mean(~nonnoise)),
             'silhouette_sample':silhouette}
    if args.algorithm=='kmeans':metrics['inertia']=float(model.inertia_)
    result=frame.copy()
    result.insert(1,'cluster_id',cluster)
    print('\nCLUSTER MEMBERSHIP (-1 means DBSCAN noise, NOT confirmed malicious):')
    print(pd.Series(cluster).value_counts().sort_index().to_string())
    label=LABELS[args.dataset]
    print('\nGROUND-TRUTH LABELS USED ONLY FOR POST-HOC ANALYSIS (not fit/feature selection):')
    print(pd.crosstab(result['cluster_id'],result[label],margins=True).to_string())
    format_table('SAMPLE CLUSTER ASSIGNMENTS',result[[IDS[args.dataset],'cluster_id',label]].head(12))
    projection=PCA(n_components=2,random_state=42).fit_transform(x)
    result.insert(2,'pca_component_1',projection[:,0])
    result.insert(3,'pca_component_2',projection[:,1])
    config={'problem':'clustering','dataset':args.dataset,'dataset_file':path.name,
            'algorithm':args.algorithm,'features':args.features,'scale':args.scale,'k':args.k,
            'eps':args.eps,'min_samples':args.min_samples,'split':'exploratory_full_unlabeled'}
    def plot(prefix):
        fig,ax=plt.subplots(figsize=(7,5))
        scatter=ax.scatter(projection[:,0],projection[:,1],c=cluster,s=8,alpha=.55,cmap='tab20')
        fig.colorbar(scatter,ax=ax,label='Cluster ID (-1 = noise)')
        ax.set(xlabel='PCA component 1',ylabel='PCA component 2',title='2D projection, PCA (for visualisation only)')
        fig.tight_layout();fig.savefig(f'{prefix}_pca_clusters.png',dpi=130);plt.close(fig)
        if args.elbow:
            inertia=[]
            ks=list(range(2,10))
            for k in ks:
                inertia.append(KMeans(n_clusters=k,n_init=5,random_state=42).fit(x).inertia_)
            fig,ax=plt.subplots(figsize=(6,4));ax.plot(ks,inertia,marker='o')
            ax.set(xlabel='Number of clusters (k)',ylabel='Inertia (within-cluster sum of squares)',title='Elbow method; subjective heuristic')
            fig.tight_layout();fig.savefig(f'{prefix}_elbow.png',dpi=130);plt.close(fig)
    save_result(args.name,config,metrics,result,plot)
    print('WARNING: Cluster IDs have no intrinsic threat meaning. Silhouette excludes DBSCAN noise.')

if __name__=='__main__':
    main()
