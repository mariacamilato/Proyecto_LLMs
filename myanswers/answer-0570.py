import pandas as pd

def contar_nulos_por_columna(df):
    return df.isnull().sum()


