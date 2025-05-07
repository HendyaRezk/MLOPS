import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from dotenv import load_dotenv

def init_mlflow():
    load_dotenv()
    dagshub.init(repo_owner='HendyaRezk', repo_name='MLOPS', mlflow=True)
    mlflow.set_tracking_uri(os.getenv('https://dagshub.com/HendyaRezk/MLOPS.mlflow'))

def load_data(data_path):
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).values.ravel()
    return X_train, y_train

def train_and_save_models(model_cfg, processed_dir):
    try:
        init_mlflow()
        mlflow.set_experiment("Titanic-Survival")
        
        X_train, y_train = load_data(processed_dir)
        
        with mlflow.start_run():
            model = RandomForestClassifier(**model_cfg['params'])
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_train)
            acc = accuracy_score(y_train, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            mlflow.log_params(model_cfg['params'])
            mlflow.log_metrics({
                'accuracy': acc,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std()
            })
            
            os.makedirs(model_cfg['output_dir'], exist_ok=True)
            model_path = f"{model_cfg['output_dir']}/{model_cfg['name']}.pkl"
            joblib.dump(model, model_path)
            
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(model_path)
            
            report = classification_report(y_train, y_pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")
            
            return model_path
            
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")