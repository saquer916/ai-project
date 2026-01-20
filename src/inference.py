"""
Real-time inference: given live stock signals, predict optimal pricing action.
"""
import pandas as pd
import numpy as np
from src.fetcher import fetch_price_history, compute_basic_signals
from config.nlp import extract_companies_from_product
from src.features import feature_engineering


class PricingAgent:
    def __init__(self, model):
        self.model = model

    def _aggregate_signals(self, sigs):
        """Helper to aggregate per-company signals into averages."""
        avg_momentum = np.mean([s.get('momentum_3d_pct', 0) or 0 for s in sigs])
        avg_pct30 = np.mean([s.get('pct_30d', 0) or 0 for s in sigs])
        max_spike = max([s.get('volume_spike', 0) or 0 for s in sigs])
        return avg_momentum, avg_pct30, max_spike

    def analyze_product(self, product_text):
        """
        End-to-end: product → companies → live signals → prediction.
        Returns dict with recommendation.
        """
        mapping = extract_companies_from_product(product_text)
        suppliers = mapping.get("suppliers", [])
        customers = mapping.get("customers", [])

        supplier_signals = []
        customer_signals = []

        for name, ticker in suppliers:
            try:
                df = fetch_price_history(ticker, period="90d")
                sig = compute_basic_signals(df)
                sig['name'] = name
                sig['ticker'] = ticker
                supplier_signals.append(sig)
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")

        for name, ticker in customers:
            try:
                df = fetch_price_history(ticker, period="90d")
                sig = compute_basic_signals(df)
                sig['name'] = name
                sig['ticker'] = ticker
                customer_signals.append(sig)
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")

        if not supplier_signals or not customer_signals:
            return {"error": "Could not fetch signals for suppliers or customers"}

        # Aggregate
        avg_supplier_momentum, avg_supplier_pct30, avg_supplier_spike = self._aggregate_signals(supplier_signals)
        avg_customer_momentum, avg_customer_pct30, avg_customer_spike = self._aggregate_signals(customer_signals)

        # Map aggregated metrics to 0–1 signals
        supplier_signal = (
            0.5 * (avg_supplier_momentum / 100.0) +
            0.5 * (avg_supplier_pct30 / 100.0)
        )
        customer_signal = (
            0.5 * (avg_customer_momentum / 100.0) +
            0.5 * (avg_customer_pct30 / 100.0)
        )

        supplier_signal = float(np.clip(supplier_signal, 0, 1))
        customer_signal = float(np.clip(customer_signal, 0, 1))

        # Create input DataFrame
        input_data = pd.DataFrame({
            'product': [product_text.lower()],
            'supplier_momentum_3d': [avg_supplier_momentum],
            'supplier_pct_30d': [avg_supplier_pct30],
            'supplier_volume_spike': [avg_supplier_spike],
            'customer_momentum_3d': [avg_customer_momentum],
            'customer_pct_30d': [avg_customer_pct30],
            'customer_volume_spike': [avg_customer_spike],
            'supplier_signal': [supplier_signal],
            'customer_signal': [customer_signal],
        })

        # Engineer features
        input_feat, feat_names = feature_engineering(input_data)
        X = input_feat[feat_names]

        # Predict
        pred_profit, confidence = self.model.predict_with_confidence(X)

        # Map to price recommendation
        if pred_profit[0] > 2:
            price_action = 5
        elif pred_profit[0] > 0:
            price_action = 3
        elif pred_profit[0] < -2:
            price_action = -2
        else:
            price_action = 0

        result = {
            'product': product_text,
            'price_recommendation_pct': float(price_action),
            'predicted_profit_impact': float(pred_profit[0]),
            'confidence': float(confidence[0]),
            'suppliers': [
                {
                    'name': s['name'],
                    'ticker': s['ticker'],
                    'pct_30d': s.get('pct_30d'),
                }
                for s in supplier_signals
            ],
            'customers': [
                {
                    'name': c['name'],
                    'ticker': c['ticker'],
                    'pct_30d': c.get('pct_30d'),
                }
                for c in customer_signals
            ],
        }

        return result
