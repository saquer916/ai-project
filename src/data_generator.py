"""Generate synthetic training dataset: historical products with stock signals and actual price outcomes. In production, replace with real historical pricing + stock data from your ERP + market data APIs."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_training_dataset(n_samples=500, seed=42):
    np.random.seed(seed)
    # Simulate product info
    product_ids = np.arange(n_samples)
    supplier_signals = np.random.rand(n_samples)
    customer_signals = np.random.rand(n_samples)
    price_change = np.random.randn(n_samples)
    profit_impact = supplier_signals * customer_signals * price_change

    # Create a DataFrame
    dataset = pd.DataFrame({
        'product_id': product_ids,
        'supplier_signal': supplier_signals,
        'customer_signal': customer_signals,
        'price_change': price_change,
        'profit_impact': profit_impact
    })

    return dataset
"""