import yfinance as yf
import pandas as pd
import numpy as np

def fetch_price_history(ticker, period="90d", interval="1d"):
    """
    Fetch historical price+volume for a ticker using yfinance.
    Returns a DataFrame with columns ['Open','High','Low','Close','Adj Close','Volume'].
    """
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, actions=False)
    if df.empty:
        raise ValueError(f"No data for ticker {ticker}")
    df = df.replace({pd.NA: None})
    return df

def compute_basic_signals(df):
    """
    Compute: 
      - last_close
      - momentum_3d_pct (3-day momentum)
      - pct_30d (30-day percent change)
      - volume_spike (today > 2x 20d average)
    """
    df = df.copy()
    last = df. iloc[-1]
    out = {}
    
    # 3-day momentum
    lookback_3 = min(len(df)-1, 3)
    if lookback_3 >= 1:
        close_3_ago = df['Close'].iloc[-1-lookback_3]
        out['momentum_3d_pct'] = (last['Close'] / close_3_ago - 1) * 100
    else:
        out['momentum_3d_pct'] = None
    
    # 30-day pct
    if len(df) > 30:
        close_30_ago = df['Close'].iloc[-31]
        out['pct_30d'] = (last['Close'] / close_30_ago - 1) * 100
    else:
        out['pct_30d'] = None
    
    # volume spike
    if len(df) >= 21:
        avg20 = df['Volume'].iloc[-21:-1].mean()
        out['volume_spike'] = bool(last['Volume'] > 2 * avg20) if avg20 and not pd.isna(avg20) else False
    else: 
        out['volume_spike'] = False
    
    out['last_close'] = last['Close']
    out['last_volume'] = int(last['Volume']) if not pd.isna(last['Volume']) else None
    
    return out