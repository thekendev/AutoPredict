import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
import joblib
import os
from Autopredictor.src.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load the data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Separate features (X) and target variable (y)
        train_x = train_data.drop(columns=[self.config.target_column])
        train_y = train_data[self.config.target_column]
        test_x = test_data.drop(columns=[self.config.target_column])
        test_y = test_data[self.config.target_column]

        # Define numerical and categorical features
        numerical_features = train_x.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = train_x.select_dtypes(include=['object']).columns.tolist()

        # Create preprocessing pipelines for numerical and categorical data
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config.preprocessor__num__imputer__strategy)),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # ColumnTransformer to apply different preprocessing to numerical and categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ]
        )

        # Create the pipeline that includes preprocessing and the classifier
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(
                C=self.config.classifier__C,
                max_iter=self.config.classifier__max_iter,
                penalty=self.config.classifier__penalty,
                solver='saga',
                random_state=42
            ))
        ])

        # Fit the pipeline
        pipeline.fit(train_x, train_y)

        # Save the trained model
        joblib.dump(pipeline, os.path.join(self.config.root_dir, self.config.model_name))
