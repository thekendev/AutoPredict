import os
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from Autopredictor.src.utils.common import read_yaml, create_directories
from Autopredictor.src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
import joblib
from src.entity.config_entity import (DataTransformationConfig)
from Autopredictor.src.logging import logger

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.preprocessor = None

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        data = data.rename(columns={'failed': 'fail'})  # Ensure column names are consistent
        train, test = train_test_split(data, test_size=0.25, random_state=42)
        train_path = self.config.root_dir / "train.csv"
        test_path = self.config.root_dir / "test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        logger.info("Split data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)
        return train, test

    def create_pipeline(self):
        numerical_features = ['year', 'price', 'mileage', 'tax', 'mpg', 'enginesize']
        categorical_features = ['model', 'transmission', 'fueltype', 'manufacturer']

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        joblib.dump(self.preprocessor, self.config.transformer_path)

    def transform_data(self):
        train, test = self.train_test_spliting()
        if self.preprocessor is None:
            raise ValueError("Preprocessor not created. Call create_pipeline() first.")
        self.preprocessor = joblib.load(self.config.transformer_path)
        X_train_transformed = self.preprocessor.fit_transform(train.drop('fail', axis=1))
        X_test_transformed = self.preprocessor.transform(test.drop('fail', axis=1))
        train_transformed_df = pd.DataFrame.sparse.from_spmatrix(X_train_transformed)
        test_transformed_df = pd.DataFrame.sparse.from_spmatrix(X_test_transformed)
        train_df = pd.concat([train[['fail']].reset_index(drop=True), train_transformed_df], axis=1)
        test_df = pd.concat([test[['fail']].reset_index(drop=True), test_transformed_df], axis=1)
        train_df.to_csv(self.config.root_dir / "train_transformed.csv", index=False)
        test_df.to_csv(self.config.root_dir / "test_transformed.csv", index=False)
