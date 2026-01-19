import xgboost as xgb
import joblib
import pandas as pd

class XGBoostModel:
    def __init__(self, params=None):
        self.model = xgb.XGBClassifier(**params) if params else xgb.XGBClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filename):
        joblib.dump(self.model, filename)

    def load(self, filename):
        self.model = joblib.load(filename)

    def feature_importance(self):
        return self.model.feature_importances_