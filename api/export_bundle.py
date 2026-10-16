"""Retrain the existing classification pipeline and persist inference artifacts."""
from pathlib import Path
import joblib
import pandas as pd
from config import load_unified_config, PipelineConfig
from preprocessing import FeatureEngineeringPipeline
from models import XGBoostChurnModel

def main():
    cfg=load_unified_config("config/config.yaml")
    import os
    data_path=os.environ.get("OFFICIAL_FEATURES_PATH","data/official/customer_features.csv")
    if not Path(data_path).is_file():
        raise FileNotFoundError(f"Official features missing: {data_path}. Run python -m api.official_data --download first.")
    df=pd.read_csv(data_path)
    features_cfg=PipelineConfig(**{**cfg.features.__dict__,"enable_smote":False})
    pipeline=FeatureEngineeringPipeline(features_cfg)
    result=pipeline.prepare_train_test(df)
    params=dict(cfg.model.params)
    params["scale_pos_weight"]=float((result.y_train==0).sum()/max(1,(result.y_train==1).sum()))
    model=XGBoostChurnModel(params=params,early_stopping_rounds=cfg.model.early_stopping_rounds)
    # Current training implementation uses holdout for early stopping.
    # Reported metrics should not be treated as unbiased production estimates.
    model.fit(result.X_train_resampled,result.y_train_resampled,result.X_test,result.y_test,verbose=False)
    path=Path("artifacts/churn_bundle.joblib")
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":model,"pipeline":pipeline,"features":result.model_features},path)
    print(f"Exported {path}")

if __name__=="__main__":
    main()
