import numpy as np
import pandas as pd
from src.data_generator import generate_training_dataset
from src.features import feature_engineering


def _rule_based_profit(df_feat: pd.DataFrame) -> np.ndarray:
    """
    Very simple rule baseline:
    - If buy_signal == 1 → use price_change
    - If sell_signal == 1 → use -price_change
    - Else 0
    """
    sign = np.where(df_feat['buy_signal'] == 1, 1, 0) - np.where(df_feat['sell_signal'] == 1, 1, 0)
    return sign * df_feat['price_change'].values


def compare_strategies(model, feature_names, n_samples: int = 500, seed: int = 123):
    """
    Generate fresh synthetic data and compare:
      - ML model
      - Simple rule baseline
    """
    df = generate_training_dataset(n_samples=n_samples, seed=seed)
    df_feat, feat_cols = feature_engineering(df)

    # Align features with model
    X = df_feat[feature_names]
    y_true = df['profit_impact'].values

    # Model strategy
    y_pred = model.predict(X)
    model_profit = y_pred.sum()

    # Rule baseline
    rule_profit_series = _rule_based_profit(df_feat)
    rule_profit = rule_profit_series.sum()

    improvement_pct = (model_profit - rule_profit) / (abs(rule_profit) + 1e-6) * 100

    result = {
        'model': {
            'total_profit': float(model_profit),
            'avg_profit_per_trade': float(model_profit / len(df)),
        },
        'rule_baseline': {
            'total_profit': float(rule_profit),
            'avg_profit_per_trade': float(rule_profit / len(df)),
        },
        'improvement_pct': float(improvement_pct),
    }

    return result
