import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, data):
        self.data = data
    
    def run_backtest(self, strategy, initial_capital=10000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = self.strategy.generate_signals(self.data)
        self.results['Portfolio Value'] = self.calculate_portfolio_value()
        return self.results

    def calculate_portfolio_value(self):
        # Portfolio value calculation logic
        return self.initial_capital + (self.results['Signal'] * self.data['Close']).cumsum()

class Strategy:
    def generate_signals(self, data):
        # Placeholder for strategy logic
        data['Signal'] = np.where(data['Close'].shift(1) < data['Close'], 1, 0)
        return data

def compare_strategies(data, strategies):
    results = {}
    for strategy in strategies:
        backtester = Backtester(data)
        results[strategy.__class__.__name__] = backtester.run_backtest(strategy)
    return results

if __name__ == "__main__":
    # Simulated market data
    dates = pd.date_range('2021-01-01', '2021-12-31')
    prices = np.random.randn(len(dates)).cumsum() + 100
    market_data = pd.DataFrame(data={'Date': dates, 'Close': prices}).set_index('Date')

    # Define strategies
    strategies = [Strategy()]

    # Compare strategies
    comparison_results = compare_strategies(market_data, strategies)
    print(comparison_results)