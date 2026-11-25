"""Offline five-model evaluation; never writes production telemetry."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score, balanced_accuracy_score, log_loss, brier_score_loss,
    confusion_matrix, silhouette_score, davies_bouldin_score, calinski_harabasz_score)
from api import module5, segmentation_service as seg
from api.scoring import score_probabilities
from model_development.config import load_unified_config
from model_development.preprocessing.splitting import DataSplitter
from demo_data.generate import BATCHES
from demo_data.validate import read_outputs, check_dataset, model_paths, segmentation_frame
from demo_data.labels import read_labels
from monitoring import drift
from monitoring.settings import digest
import yaml

ROOT = Path(__file__).resolve().parents[1]

class TrustedCheckpointLoader(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("models.logistic_regression", "models.random_forest", "models.focal_loss"):
            module = "model_development." + module
        return super().find_class(module, name)

def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pred = p >= .5
    both = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(ROC_AUC=float(roc_auc_score(y,p)) if both else None,
        PR_AUC=float(average_precision_score(y,p)) if both else None,
        F1_Score=float(f1_score(y,pred,zero_division=0)),
        Precision=float(precision_score(y,pred,zero_division=0)),
        Recall=float(recall_score(y,pred,zero_division=0)),
        Accuracy=float(accuracy_score(y,pred)),
        Balanced_Accuracy=float(balanced_accuracy_score(y,pred)) if both else None,
        Log_Loss=float(log_loss(y,p,labels=[0,1])), Brier=float(brier_score_loss(y,p)),
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        prevalence=float(y.mean()), predicted_positive_rate=float(pred.mean()))

def checked(p, n):
    p=np.asarray(p,dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError("Invalid probability vector")
    return p

def cluster_quality(z, labels, centers, shares, p95, cfg):
    d=np.linalg.norm(z-centers[labels],axis=1)
    counts=np.bincount(labels,minlength=len(centers))
    active=np.count_nonzero(counts)
    valid=2<=active<len(z)
    distance_ref=drift.categorical_reference(np.repeat(np.arange(len(shares)),shares))
    distances=np.linalg.norm(z[:,None,:]-centers[None,:,:],axis=2)
    ordered=np.sort(distances,axis=1)
    return dict(silhouette=float(silhouette_score(z,labels)) if valid else None,
        dbi=float(davies_bouldin_score(z,labels)) if valid else None,
        chi=float(calinski_harabasz_score(z,labels)) if valid else None,
        mean_distance=float(d.mean()), p95_distance=float(np.quantile(d,.95)),
        distortion=float(np.mean(d*d)),
        outlier_share=float(np.mean(d>p95[labels])) if np.isfinite(p95[labels]).all() else None,
        margin=float(np.mean(ordered[:,1]-ordered[:,0])),
        active_clusters=int(active), counts=counts.tolist(),
        cluster_js=drift.compare(labels,distance_ref,cfg,name="cluster")["score"])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/demo")
    parser.add_argument("--labels",type=Path)
    parser.add_argument("--export-dir",type=Path,required=True)
    args=parser.parse_args()
    out=args.export_dir; out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/"config/monitoring.yaml").read_text())
    model=module5.load(); bundle=seg.load()
    if model is None or bundle is None: raise ValueError("Missing validated serving artifacts")
    manifest, frame, examples=read_outputs(args.data_dir)
    if manifest["input_columns"]!=model.input_columns or manifest["input_dtypes"]!=model.manifest["input_dtypes"]:
        raise ValueError("Schema mismatch")
    for name,path in model_paths().items():
        if module5.file_hash(path)!=manifest["models"][name]["sha256"]: raise ValueError("Artifact mismatch: "+name)
    if manifest["inference"]["status"]!="passed": raise ValueError("Validate demo first")
    source_ids=set(manifest["source_customer_id_by_demo_id"].values())
    if len(source_ids)!=manifest["N_base"]:
        source_ids=pd.read_csv(module5.DATA_PATH,usecols=["customer_id"]).customer_id
    check_dataset(frame,examples,manifest,source_ids)
    frame=frame.reset_index(drop=True)
    if len(frame)!=10000 or not frame.customer_id.is_unique or set(frame.batch_id)!=set(BATCHES) or not (frame.batch_id.value_counts()==2000).all():
        raise ValueError("Expected five disjoint cohorts of 2000")
    dataset_hash=manifest["output_sha256"]["customer_features.csv"]
    labels,meta=read_labels(args.labels,frame,dataset_hash) if args.labels else (None,None)
    y=labels.loc[frame.customer_id,"churn"].to_numpy() if labels is not None else None
    raw=frame[model.input_columns]
    original=pd.read_csv(module5.DATA_PATH)
    splitcfg=load_unified_config(module5.CONFIG_PATH).features
    split=DataSplitter(test_size=splitcfg.test_size,val_size=splitcfg.val_size,
        stratify=splitcfg.stratify,random_state=splitcfg.random_state).train_val_test_split(original)
    train=split.train_df; test=split.test_df
    ztrain=model.transform(train[model.input_columns]); ztest=model.transform(test[model.input_columns])
    z=model.transform(raw); features=model.manifest["features"]
    if list(z.columns)!=features: raise ValueError("Feature order mismatch")
    expected=pd.read_csv(module5.REPORT_PATH).set_index("Model")
    base_ref=json.loads(module5.REFERENCE_PATH.read_text())
    masks={b:frame.batch_id.eq(b).to_numpy() for b in BATCHES}
    normal=np.logical_or.reduce([masks[b] for b in BATCHES[:3]])
    rows=[]; clusters=[]; inputs=[]; validation=[]; blockers={}
    profiles=model.profiles(raw)
    for b,mask in masks.items():
        r=drift.evaluate(profiles.loc[mask],base_ref,cfg)
        inputs.extend(dict(batch=b,space="classifier",**x) for x in r["features"])
    for name in ("XGBOOST","LR","RF"):
        try:
            if name=="XGBOOST":
                checkpoint=module5.CHECKPOINT_PATH
                estimator=model.classifier
            else:
                checkpoint=ROOT/"model_development/models/saved"/(name.lower()+"_churn.pkl")
                with checkpoint.open("rb") as f: wrapper=TrustedCheckpointLoader(f).load()
                if list(wrapper.feature_names_)!=features: raise ValueError("Checkpoint feature order mismatch")
                estimator=wrapper.model
            if list(estimator.classes_)!=[0,1]: raise ValueError("Class mapping mismatch")
            def predict(x): return checked(estimator.predict_proba(x)[:,1],len(x))
            actual=metrics(test.churn.to_numpy(),predict(ztest))
            keys=("ROC_AUC","PR_AUC","F1_Score","Precision","Recall","Accuracy","TP","FP","TN","FN")
            mismatch={k:[actual[k],float(expected.loc[name,k])] for k in keys if abs(actual[k]-float(expected.loc[name,k]))>5e-5}
            validation.append(dict(algorithm=name,checkpoint_sha256=module5.file_hash(checkpoint),actual=actual,mismatches=mismatch))
            if mismatch: raise ValueError("Holdout reproduction failed: "+str(mismatch))
            prob=predict(z); ref=drift.numeric_reference(predict(ztrain),cfg["histogram_bins"])
            baseline=metrics(y[normal],prob[normal]) if y is not None else {}
            model_id=digest([name,module5.file_hash(checkpoint),model.manifest["preprocessor_sha256"],ref,.5])
            for b,mask in masks.items():
                m=metrics(y[mask],prob[mask]) if y is not None else {}
                rows.append(dict(algorithm=name,model_id=model_id,batch=b,N=int(mask.sum()),status="ready" if y is not None else "awaiting_labels",
                    probability_psi=drift.compare(prob[mask],ref,cfg,name="probability")["score"],**m,
                    **{"baseline_"+k:v for k,v in baseline.items() if k in ("ROC_AUC","PR_AUC","F1_Score","Recall")},
                    **{"drop_"+k:baseline[k]-m[k] for k in ("ROC_AUC","PR_AUC","F1_Score","Recall") if b not in BATCHES[:3] and k in m and m[k] is not None and baseline[k] is not None}))
        except Exception as e:
            blockers[name]=str(e)
    sf=seg.feature_frame(segmentation_frame(frame))
    official=seg.feature_frame(pd.read_csv(seg.DATA_PATH))
    enc=bundle["preprocessor"].transform(sf)
    refenc=bundle["preprocessor"].transform(official)
    segrefs={name:(drift.numeric_reference(official[name],cfg["histogram_bins"]) if pd.api.types.is_numeric_dtype(official[name]) else drift.categorical_reference(official[name])) for name in official}
    for b,mask in masks.items():
        inputs.extend(dict(batch=b,space="clustering",**drift.compare(sf.loc[mask,name],ref,cfg,name=name)) for name,ref in segrefs.items())
    for name in seg.ALGORITHMS:
        centers=bundle["kmeans"].cluster_centers_ if name=="kmeans" else bundle["hierarchical_centers"]
        assigned=seg.predict_clusters(bundle,sf,name)
        refassigned=seg.predict_clusters(bundle,official,name)
        d=np.linalg.norm(refenc-centers[refassigned],axis=1)
        p95=np.array([np.quantile(d[refassigned==k],.95) if (refassigned==k).any() else np.nan for k in range(len(centers))])
        shares=np.bincount(refassigned,minlength=len(centers))
        model_id=digest([name,manifest["models"]["clustering_bundle"]["sha256"],"fixed_assignment_v1"])
        group=[]
        for b,mask in masks.items():
            q=cluster_quality(enc[mask],assigned[mask],centers,shares,p95,cfg)
            group.append(dict(algorithm=name,model_id=model_id,batch=b,N=int(mask.sum()),assignment_method="kmeans_predict" if name=="kmeans" else "ward_nearest_centroid_surrogate",**q))
        for row in group:
            for key in ("silhouette","dbi","chi","p95_distance","outlier_share"):
                vals=[r[key] for r in group[:3] if r[key] is not None]
                base=float(np.median(vals)) if vals else None
                row["baseline_"+key]=base
                if row["batch"] not in BATCHES[:3] and base is not None and row[key] is not None:
                    row["drop_"+key]=(base-row[key]) if key in ("silhouette","chi") else (row[key]-base)
        clusters.extend(group)
    for filename,records in (("classifier_metrics",rows),("clustering_metrics",clusters),("input_drift",inputs)):
        pd.DataFrame(records).to_csv(out/(filename+".csv"),index=False)
        (out/(filename+".json")).write_text(json.dumps(records,indent=2,allow_nan=False))
    result=dict(dataset_sha256=dataset_hash,label_metadata=meta,blockers=blockers,artifact_validation=validation,
        classifier_results=len(rows),clustering_results=len(clusters),status="complete" if len(rows)==15 and len(clusters)==10 else "partial")
    (out/"verification.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False))
    if result["status"]!="complete": raise SystemExit(1)

if __name__=="__main__": main()
