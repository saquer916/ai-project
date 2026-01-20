import pandas as pd
import numpy as np


def _ensure_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure supplier_signal and customer_signal exist, deriving them if needed."""
    df = df.copy()

    has_supplier_signal = 'supplier_signal' in df.columns
    has_customer_signal = 'customer_signal' in df.columns

    if not has_supplier_signal:
        if 'supplier_momentum_3d' in df.columns and 'supplier_pct_30d' in df.columns:
            df['supplier_signal'] = (
                0.5 * df['supplier_momentum_3d'].fillna(0) / 100
                + 0.5 * df['supplier_pct_30d'].fillna(0) / 100
            ).clip(0, 1)
        else:
            df['supplier_signal'] = 0.5

    if not has_customer_signal:
        if 'customer_momentum_3d' in df.columns and 'customer_pct_30d' in df.columns:
            df['customer_signal'] = (
                0.5 * df['customer_momentum_3d'].fillna(0) / 100
                + 0.5 * df['customer_pct_30d'].fillna(0) / 100
            ).clip(0, 1)
        else:
            df['customer_signal'] = 0.5

    return df


def feature_engineering(df: pd.DataFrame):
    df = _ensure_signals(df)

    # Buy / sell signals
    df['buy_signal'] = ((df['customer_signal'] > 0.6) & (df['supplier_signal'] < 0.4)).astype(int)
    df['sell_signal'] = ((df['customer_signal'] < 0.4) & (df['supplier_signal'] > 0.6)).astype(int)

    # Derived signal features
    df['signal_strength'] = (df['buy_signal'] - df['sell_signal']).abs()
    df['direction'] = df['buy_signal'] - df['sell_signal']

    # Product category
    if 'product_category' not in df.columns:
        categories = ['steel components', 'car parts', 'electronics', 'chemicals', 'textiles']
        df['product_category'] = np.random.choice(categories, size=len(df))

    # FIXED: Create dummies for ALL possible categories explicitly
    all_categories = ['steel components', 'car parts', 'electronics', 'chemicals', 'textiles']
    
    # Create one-hot encoding manually to ensure ALL categories exist
    for cat in all_categories:
        col_name = f'category_{cat}'
        df[col_name] = (df['product_category'] == cat).astype(int)

    # Final features - EXACTLY 11 features in fixed order
    feature_cols = [
        'supplier_signal',
        'customer_signal',
        'buy_signal',
        'sell_signal',
        'signal_strength',
        'direction',
        'category_steel components',
        'category_car parts', 
        'category_electronics',
        'category_chemicals',
        'category_textiles',
    ]

    return df, feature_cols
