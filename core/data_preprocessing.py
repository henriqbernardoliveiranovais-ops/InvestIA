import pandas as pd

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(window=20).std()
    df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
    return df

def prepare_features(df: pd.DataFrame, days_ahead: int = 5):
    df = calculate_technical_indicators(df)
    df["Target"] = df["Close"].shift(-days_ahead)
    df_clean = df.dropna()
    features = ["Close","SMA_5","SMA_20","SMA_50","Volatility","Volume_MA","Daily_Return"]
    X = df_clean[features]
    y = df_clean["Target"]
    return X, y, df_clean
