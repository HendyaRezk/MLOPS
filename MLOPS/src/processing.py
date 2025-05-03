import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_and_save(data_cfg):
    train_df = pd.read_csv(data_cfg.raw_train_path)
    test_df = pd.read_csv(data_cfg.raw_test_path)

    features = ['Sex', 'Embarked', 'Age', 'Fare']
    target = 'Survived'

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]

    cat_cols = ['Sex', 'Embarked']
    num_cols = ['Age', 'Fare']

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, cat_cols),
        ('num', num_pipeline, num_cols)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_df = pd.DataFrame(
        X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed
    )
    X_test_df = pd.DataFrame(
        X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed
    )

    X_train_df['Survived'] = y_train.reset_index(drop=True)

    os.makedirs(data_cfg.processed_dir, exist_ok=True)

    X_train_df.to_csv(data_cfg.train_processed, index=False)
    X_test_df.to_csv(data_cfg.test_processed, index=False)

    print(f"Preprocessing complete. Files saved in: {data_cfg.processed_dir}")
