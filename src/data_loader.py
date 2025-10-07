from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import os

def load_20newsgroups(subset='train', remove=('headers','footers','quotes')):
    data = fetch_20newsgroups(subset=subset, remove=remove)
    df = pd.DataFrame({'text': data.data, 'target': data.target})
    df['target_name'] = df['target'].apply(lambda i: data.target_names[i])
    return df

def load_csv(path, text_col='text', label_col=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"{text_col} not found in CSV columns")
    if label_col and label_col not in df.columns:
        raise ValueError(f"{label_col} not found in CSV columns")
    return df
