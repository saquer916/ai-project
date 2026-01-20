import pandas as pd
import numpy as np

def feature_engineering(df):
    df = df.copy()

    # ----------------------------
    # 1. Create buy/sell signals based on supplier & customer signals
    # ----------------------------
    # Example: buy if customer_signal high and supplier_signal low, sell if opposite
    df['buy_signal'] = ((df['customer_signal'] > 0.6) & (df['supplier_signal'] < 0.4)).astype(int)
    df['sell_signal'] = ((df['customer_signal'] < 0.4) & (df['supplier_signal'] > 0.6)).astype(int)

    # ----------------------------
    # 2. Derived signal features
    # ----------------------------
    df['signal_strength'] = (df['buy_signal'] - df['sell_signal']).abs()
    df['direction'] = df['buy_signal'] - df['sell_signal']

    # ----------------------------
    # 3. Optionally create a synthetic product category
    # ----------------------------
    if 'product_category' not in df.columns:
        categories = ['steel components', 'car parts', 'electronics', 'chemicals', 'textiles']
        df['product_category'] = np.random.choice(categories, size=len(df))

    category_dummies = pd.get_dummies(df['product_category'], prefix='category')
    df = pd.concat([df, category_dummies], axis=1)

    # ----------------------------
    # 4. Final feature list
    # ----------------------------
    feature_cols = [
        'supplier_signal',
        'customer_signal',
        'buy_signal',
        'sell_signal',
        'signal_strength',
        'direction'
    ] + list(category_dummies.columns)

    return df, feature_cols
