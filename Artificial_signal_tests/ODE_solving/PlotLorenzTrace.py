#%% imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% load data to dataframe
lorenz_data = pd.read_csv('Lorenz_maps/SmallEnsemble_Lorenz.csv')
lorenz_data_description = lorenz_data.describe()

#%% process columns
columns = lorenz_data_description.columns
split_col = np.array([column.split('-') for column in columns], dtype=object)
axis = sorted(set(split_col[:, 0]))
params = list(set(split_col[:, 1]))
header = pd.MultiIndex.from_product([params, axis], names=['params', 'axis'])
lorenz_data.columns = header

#%% go through the data traces and create 3d plots of the axis
for param_set in params:
    sub_df = lorenz_data.xs(param_set, level='params', axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x, y, z = sub_df.to_numpy().T
    index = np.arange(0, 1,1/len(x))
    p = ax.scatter(x, y, z, label=param_set, c=index, marker='o', edgecolor=None, s=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.colorbar(p)
    fig.suptitle('Lorenz Trace Plot')
    fig.tight_layout()
    ax.legend()
    plt.show()