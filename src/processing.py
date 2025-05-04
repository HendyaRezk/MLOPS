import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import yaml

def preprocess_and_save():
    # Load config
    cfg = yaml.safe_load(open("params/base.yaml"))
    data_cfg = cfg["data"]

    # Load raw data
    train_df = pd.read_csv(data_cfg["raw_train_path"])
    test_df = pd.read_csv(data_cfg["raw_test_path"])

    # Define features and target
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'

    # Split into categorical and numerical
    cat_cols = ['Sex', 'Embarked']
    num_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols)
    ])

    # Process the features
    X_train = preprocessor.fit_transform(train_df[features])
    X_test = preprocessor.transform(test_df[features])
    y_train = train_df[target]

    # ✅ Ensure the output directory exists
    os.makedirs(data_cfg["processed_dir"], exist_ok=True)
    
    # Save the processed feature and target datasets
    pd.DataFrame(X_train).to_csv(os.path.join(data_cfg["processed_dir"], "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(data_cfg["processed_dir"], "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(data_cfg["processed_dir"], "y_train.csv"), index=False)

    print("Preprocessing complete and data saved successfully!")

if __name__ == "__main__":
    preprocess_and_save()
