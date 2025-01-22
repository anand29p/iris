import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    """Load Iris dataset from CSV"""
    return pd.read_csv("src/data/Iris.csv")

def preprocess(df):
    """Preprocess Iris dataset"""
    df = df.drop(columns=["Id"])
    X = df.drop(columns=["Species"])
    y = df["Species"]
    return train_test_split(X, y, test_size=0.2, random_state=42)