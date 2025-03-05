#%% imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os

#%% read data
print(os.getcwd())
Lorenz_metrics = pd.read_csv('./Artificial_signal_tests/metrics/mock_signals/simple_mock_signal_metrics.csv')

#%% get some info of the frame
columns = list(Lorenz_metrics.columns)
index = list(Lorenz_metrics.index)
data_cols = columns[4:]
info_cols = columns[:4]
Lorenz_data = Lorenz_metrics[data_cols]
Lorenz_info = Lorenz_metrics[info_cols]


#%% Bar graphs per metric
colors = ['green', 'green', 'green', 'blue', 'blue', 'blue', 'red', 'red', 'red', 'red', 'red',
           'red', 'limegreen', 'limegreen', 'limegreen', 'lime', 'lime', 'lime', 'turquoise', 'turquoise', 'aquamarine',
           'aquamarine', 'gold', 'gold', 'gold',
           'gold', 'violet', 'violet', 'violet']
metrics = Lorenz_info[['metric']].to_numpy()
fig, axes = plt.subplots(len(metrics), 1, sharex=True, sharey=False, figsize=(15, 12))
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    data = Lorenz_data.iloc[idx, :]
    sns.barplot(data=data,
                ax=ax,
                width=0.8,
                palette=colors)
    ax.set_title(metric)
    ax.set_ylabel('')
ax.tick_params(axis='x', rotation=80)
fig.tight_layout(pad=0.3)
plt.show()
