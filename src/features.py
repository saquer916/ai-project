import pandas as pd


def feature_engineering(stock_signals, supplier_risk_scores, customer_opportunity_scores, product_categories):
    \"\"\" Convert stock signals into machine learning features. \"\"\"  

    # Create signal strength and direction indicators
    stock_signals['signal_strength'] = (stock_signals['buy_signal'] - stock_signals['sell_signal']).abs()
    stock_signals['direction'] = stock_signals['buy_signal'] - stock_signals['sell_signal']

    # Combine supplier and customer scores into features
    stock_signals['supplier_risk_score'] = supplier_risk_scores
    stock_signals['customer_opportunity_score'] = customer_opportunity_scores

    # One-hot encoding of product categories
    category_dummies = pd.get_dummies(product_categories, prefix='category')
    stock_signals = pd.concat([stock_signals, category_dummies], axis=1)

    return stock_signals
