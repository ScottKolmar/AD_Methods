import pandas as pd
import numpy as np

###################
# UTILS
##################

def drop_infs(df):
    """
    Drops columns which contain infinite values in a dataframe.

    Parameters:
    df (dataframe): Dataframe to drop infinite values. (Default = 600)

    Returns:
        df (dataframe): Dataframe with infinite values dropped.
    """
    cols = [x for x in df.columns if df[x].dtype == 'object']
    df = df.drop(columns=cols)
    df[df.columns] = df[df.columns].astype(float)
    df = df.fillna(0.0).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how='any')
    return df