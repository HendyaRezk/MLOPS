import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import yaml

def preprocess_and_save():
    cfg = yaml.safe_load(open("params/base.yaml"))
    data_cfg = cfg["data"]

    train_df = pd.read_csv(data_cfg["raw_train_path"])
    test_df = pd.read_csv(data_cfg["raw_test_path"])

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'

    cat_cols = ['Sex', 'Embarked']
    num_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']

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

    X_train = preprocessor.fit_transform(train_df[features])
    X_test = preprocessor.transform(test_df[features])
    y_train = train_df[target]

    os.makedirs(data_cfg["processed_dir"], exist_ok=True)
    
    pd.DataFrame(X_train).to_csv(os.path.join(data_cfg["processed_dir"], "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(data_cfg["processed_dir"], "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(data_cfg["processed_dir"], "y_train.csv"), index=False)

if __name__ == "__main__":
    preprocess_and_save()
