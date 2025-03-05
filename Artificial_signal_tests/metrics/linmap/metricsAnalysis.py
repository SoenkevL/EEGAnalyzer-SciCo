## Imports
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os


def handle_missing_data(df, replacement_value=0, nan_threshold=0.3):
    """
    Handle missing data in the dataframe by dropping rows with NaN values
    exceeding a threshold or replacing them with a specified value.

    Args:
        df (pd.DataFrame): The input dataframe.
        replacement_value (int, float): Value to replace NaNs with.
        nan_threshold (float): Maximum proportion of NaNs allowed per row;
                               rows exceeding this will be removed.

    Returns:
        pd.DataFrame: Dataframe after handling missing values.
    """
    # Calculate percentage of NaNs per row
    nan_percentage = df.isnull().mean(axis=1)

    # Print rows with NaNs and their percentage
    rows_with_nans = df[nan_percentage > 0]
    # print("Rows with NaN values and their percentage:")
    # print(rows_with_nans.assign(nan_percentage=nan_percentage * 100))

    # Remove rows with NaN percentage above the threshold
    df_cleaned = df[nan_percentage <= nan_threshold]

    # Replace remaining NaNs with the replacement value
    df_cleaned = df_cleaned.fillna(replacement_value)

    return df_cleaned


def load_metrics(file_path):
    ## Load dataframe
    df = pd.read_csv(file_path, header=None, index_col=None)
    df = df.round(2)
    df = df.iloc[:, 3:]
    df = df.transpose()
    df.iloc[0, 0] = 'r'
    df.columns = df.iloc[0]
    df = df.iloc[1:, :]
    df = df.set_index('r')
    return df

def load_log_map_data(file_path, offset=50):
    '''
    function expects a filepath to a csv with log_map_data where the columns are the r values and rows
    the corresponding timeseries for each r
    :param file_path: path to csv with log_map_data
    :param offset: offset to remove initial values of the timetrace
    :return: pandas DataFrame with the data
    '''
    df = pd.read_csv(file_path)
    df = df.iloc[offset:, :]
    return df


def normalize_by_max(x, bound=1):
    return (x / max(abs(x))) * bound


def normalize(df, bound=1):
    normalize_by_max_partial = partial(normalize_by_max, bound=bound)
    return df.apply(func=normalize_by_max_partial, by_row=False)


def clutteredPlot(df, block=False):
    sns.lineplot(data=normalize(df)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Metrics Across r')
    plt.xlabel('r (growth factor)')
    plt.ylabel('Metric Value')
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.show(block=block)

def scatterPlotPerMetricWithNanHandling(df):
    size_df = df.copy()
    size_df = size_df.isna()
    # change the values from bool to string to have sensible labels within the plots later
    computable_df = size_df.map(lambda x: 'Not computable' if x else 'Value')
    df = df.fillna(0)
    columns = df.columns
    for column in columns:
        fig, ax = plt.subplots()
        name_sizes = {'Not computable': 0, 'Value': 10}  # Map sizes based on boolean values
        sns.scatterplot(x=df.index, y=normalize(df[column]).to_numpy(), size=computable_df[column], sizes=name_sizes, color='blue', ax=ax)
        x_ticks = [float(index) for i, index in enumerate(df.index) if i % 10 == 0]
        x_ticklabels = [str(x_ticks) if i % 2 == 0 else "" for i, x_ticks in enumerate(x_ticks)]
        ax.set_xticks(x_ticks, labels=x_ticklabels)
        ax.tick_params(axis='x', labelrotation=45)
        plt.title(column)
        plt.xlabel('r (growth factor)')
        plt.ylabel(f'{column} Value')
        plt.tight_layout()
        plt.show()

def plot_logistic_map_bifurcation(df):
    """
    Plot a bifurcation diagram with columns of the dataframe as the x-axis and corresponding values as y-axis.

    Args:
        df (pd.DataFrame): Dataframe where columns represent x-axis, and values in columns represent y-axis data points.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    for column in df.columns:
            sns.scatterplot(x=[float(column)] * len(df), y=df[column], ax=ax, alpha=0.5, s=10, color='blue')
    # show only every 10th x_tick and rotate them by 45 degrees
    x_ticks = [float(column) for i, column in enumerate(df.columns) if i % 10 == 0]
    x_ticklabels = [str(x_ticks) if i%2==0 else "" for i, x_ticks in enumerate(x_ticks)]
    ax.set_xticks(x_ticks, labels=x_ticklabels)
    ax.tick_params(axis='x', labelrotation=45)
    # set (axis-)labels
    ax.set_title('Logistic Map Bifurcation')
    ax.set_xlabel('Columns (r values)')
    ax.set_ylabel('Values')
    plt.tight_layout()
    plt.show()

def plot_logistic_map_bifurcation_with_coloring(logmap_df, metrics_df, metrics):
    """
    Plot a bifurcation diagram with columns of the dataframe as the x-axis and corresponding values as y-axis.
    Additonally the points in the diagram should be colored according to the metric.
    Args:
        logmap_df (pd.DataFrame): Dataframe where columns represent x-axis, and values in columns represent y-axis data points.

    Returns:
        None
    """
    if not isinstance(metrics, list):
        metrics = [metrics]
    for metric in metrics:
        ## creating a color mapping for the individual r using the metrics_df
        # first, we need to extract the correct metric into a series, where the index is the r
        metric_is_nan = metrics_df[metric].isna()
        metrics = metrics_df[metric].fillna(0)
        metric_max, metric_min = metrics.max(), metrics.min()
        metrics = normalize(metrics).to_numpy()
        r_values = metrics_df.index
        # then we create a mapping for all r values to a rgba value
        cmap = mpl.colormaps.get_cmap('viridis')
        color_mapping = {float(r):[cmap(metric)] if not isna else [[0., 0., 0., 0.8]] for r, metric, isna in zip(r_values, metrics, metric_is_nan)}
        ## plotting the bifurcation diagram
        fig, ax = plt.subplots()
        for column in logmap_df.columns:
            current_r = float(column)
            p = sns.scatterplot(x=[current_r] * len(logmap_df), y=logmap_df[column], ax=ax, alpha=0.5, s=10, color=color_mapping[current_r])
        # set a colobar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=metric_min, vmax=metric_max))
        fig.colorbar(sm, ax=ax, label=metric)
        # fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=None), ax=ax, label=metric)
        # show only every 10th x_tick and rotate them by 45 degrees
        x_ticks = [float(column) for i, column in enumerate(logmap_df.columns) if i % 10 == 0]
        x_ticklabels = [str(x_ticks) if i%2==0 else "" for i, x_ticks in enumerate(x_ticks)]
        ax.set_xticks(x_ticks, labels=x_ticklabels)
        ax.tick_params(axis='x', labelrotation=45)
        # set (axis-)labels
        ax.set_title('Logistic Map Bifurcation')
        ax.set_xlabel('Columns (r values)')
        ax.set_ylabel('Values')
        plt.tight_layout()
        plt.show()

def main():
    metrics_df = load_metrics('Long/Rounded/metrics.csv')
    # scatterPlotPerMetricWithNanHandling(metrics_df)
    # print(os.getcwd())
    log_map_df = load_log_map_data('/home/soenkevanloh/Documents/EEGAnalyzer/Artificial_signal_tests/csv/Round_Transpose_Logistic_map_Long.csv', offset=100)
    # plot_logistic_map_bifurcation(log_map_df)
    plot_logistic_map_bifurcation_with_coloring(log_map_df, metrics_df,
                                                ['largest_lyapunov_exponent', 'lempel_ziv_complexity',
                                                 'permutation_entropy', 'multiscale_entropy', 'fractal_dimension_higuchi_k-10'])
    return 0

if __name__ == '__main__':
    main()