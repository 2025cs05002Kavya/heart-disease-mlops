import pandas as pd

def clean_data(df):

    df = df.dropna()

    return df