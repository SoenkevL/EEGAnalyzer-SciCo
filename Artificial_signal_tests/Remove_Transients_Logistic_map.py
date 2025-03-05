import numpy as np
import pandas as pd


df = pd.read_csv("csv/Logistic_map_Long.csv", header=None, index_col=None)
df = df.iloc[:350, :]
df = df.transpose()
df.to_csv("Transpose_Logistic_map_Long.csv", header=False, index=False)
df = df.round(3)
df.to_csv("Round_Transpose_Logistic_map_Long.csv", header=False, index=False)