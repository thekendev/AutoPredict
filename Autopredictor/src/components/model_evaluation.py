
import os
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
)
from urllib.parse import urlparse
from Autopredictor.src.utils.common import save_json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from pathlib import Path
from src.entity.config_entity import (ModelEvaluationConfig)



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred, pred_proba):
        conf_matrix = confusion_matrix(actual, pred)
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred_proba)
        return conf_matrix, accuracy, precision, recall, f1, roc_auc

    def save_results(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]
        
        predicted_labels = model.predict(test_x)
        predicted_probabilities = model.predict_proba(test_x)[:, 1]

        (conf_matrix, accuracy, precision, recall, f1, roc_auc) = self.eval_metrics(test_y, predicted_labels, predicted_probabilities)
        
        # Saving metrics as local
        scores = {
            "confusion_matrix": conf_matrix.tolist(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
        save_json(path=Path(self.config.metric_file_name), data=scores)