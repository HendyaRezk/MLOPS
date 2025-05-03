import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_save_models(model_cfg, processed_dir):
    df = pd.read_csv(os.path.join(processed_dir, 'train_processed.csv'))
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_cfg.type == "RandomForestClassifier":
        model = RandomForestClassifier(**model_cfg.params)
    elif model_cfg.type == "LogisticRegression":
        model = LogisticRegression(**model_cfg.params)
    else:
        raise ValueError(f"Unsupported model type: {model_cfg.type}")

    os.makedirs(model_cfg.output_dir, exist_ok=True)
    model_path = os.path.join(model_cfg.output_dir, f"{model_cfg.name}.pkl")

    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}. Skipping training.")
        model = joblib.load(model_path)  
        preds = model.predict(X_val)  
        acc = accuracy_score(y_val, preds)  
        print(f"✅ Loaded model '{model_cfg.name}' accuracy on validation set: {acc:.4f}")
        print(classification_report(y_val, preds))  
        return

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"{model_cfg.name} accuracy: {acc:.4f}")

    joblib.dump(model, model_path)
    print(f"{model_cfg.name} saved to {model_path}")
