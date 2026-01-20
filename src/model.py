import xgboost as xgb
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class XGBoostModel:
    def __init__(self, params=None):
        # Regressor because your target is continuous
        self.model = xgb.XGBRegressor(**params) if params else xgb.XGBRegressor()
        self.features = None
        self.metadata = {}

    def train(self, X, y, test_size=0.2, random_state=42):
        self.features = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model.fit(X_train, y_train)

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        metrics = {
            'train_rmse': mean_squared_error(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_rmse': mean_squared_error(y_test, y_test_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test)
        }

        self.metadata['metrics'] = metrics
        return metrics

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filename):
        joblib.dump(self.model, filename)  # filename must include .pkl

    def load(self, filename):
        self.model = joblib.load(filename)

    def feature_importance(self):
        return pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values(by='importance', ascending=False)
