#%% imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os

#%% read data
print(os.getcwd())
Lorenz_metrics = pd.read_csv('SmallEnsemble_metrics.csv')

#%% get some info of the frame
columns = list(Lorenz_metrics.columns)
index = list(Lorenz_metrics.index)
data_cols = columns[4:]
info_cols = columns[:4]
Lorenz_data = Lorenz_metrics[data_cols]
Lorenz_info = Lorenz_metrics[info_cols]


#%% Bar graphs per metric
metrics = Lorenz_info[['metric']].to_numpy()
fig, axes = plt.subplots(len(metrics), 1, sharex=True, sharey=False, figsize=(15, 12))
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    data = Lorenz_data.iloc[idx, :]
    sns.barplot(data=data,
                ax=ax,
                width=0.8)
    ax.set_title(metric)
ax.tick_params(axis='x', rotation=80)
fig.tight_layout(pad=0.3)
plt.show()
