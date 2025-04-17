import datetime as dt
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.traceback import install


def log_map(x, r):
    return r*x*(1-x)

def init(x=None, r=None):
    # print(x, r)
    if not x or x >= 1 or x <= 0:
        x = float(input('PLease assign a value to x, otherwise the logistic map cant be computed, needs to be (0,1)'))
    if not r:
        r = float(input('Please assign a value to r, otherwise the logistic map cant be computed'))
    return x, r

def create_log_map_trace(x_init, r_init, length):
    # creates a trace for the logistic map starting at x_init with value r and
    # over the length length
    x_ver, r = init(x_init, r_init)
    arr = np.zeros(length)
    arr[0] = x_ver
    for i in np.arange(1,length,1):
        arr[i] = log_map(arr[i-1], r)
    # print(arr)
    return arr


def check_csv_path(csv_path):
    """
    Verify or construct a valid CSV file path from the input parameter and return the path.

    This function checks if the provided path to a CSV file is valid. If the path is incomplete
    or refers to a directory, it generates an ad hoc filename based on the current date and time.
    If no path is provided, the function constructs a CSV file path in the current working directory
    with an ad hoc name.

    :param csv_path: The provided path to a CSV file or a directory. If `None`,
                     a new path will be generated in the current working directory.
    :type csv_path: str or None

    :return: A complete and valid CSV file path. If the provided path is invalid,
             the function returns 0.
    :rtype: str or int
    """
    # is a path given?
    current_file_path = Path(__file__).parent
    pwd = os.getcwd()
    if csv_path:
        basename, dirname = os.path.split(csv_path)
        # check if the csv path is given to a folder or to an actual csv
        if dirname.endswith('.csv'):
            # most likely a valid path to a csv
            if basename:
                return os.path.join(basename, dirname)
            else:
                return os.path.join(current_file_path, dirname)
        elif '.' in dirname:
            print('path is not csv but has an extension, this is invalid')
            return 0
        # create a name for the csv as only a path is given
        current_datetime = dt.datetime.now()
        ad_hoc_name = f'data__{current_datetime:%Y_%m_%d_%H_%M_%S}.csv'
        return os.path.join(basename, dirname, ad_hoc_name)
    else:
        current_datetime = dt.datetime.now()
        ad_hoc_name = f'data__{current_datetime:%y_%m_%d_%H_%M_%S}.csv'
        return os.path.join(pwd, ad_hoc_name)


def test_csv_path():
    paths =[
            'test.csv',
            './test/text.txt',
            '//home/soenkevanloh/Documents/ODE_solving/test',
            '/home/soenkevanloh/Documents/ODE_solving/test/test.csv'
            ]
    for path in paths:
        print(check_csv_path(path))


def save_df_to_csv(df, csv, append=True):
    # saves a dataframe to a csv path and tries to append if a csv at that path
    # already exists, otherwise creates a new csv
    if os.path.exists(csv):
        if append:
            temp_df = pd.read_csv(csv, header=None)
            comb_df = pd.concat([temp_df, df], ignore_index=True)  # Append rows, not columns
        else:
            comb_df = df
    else:
        comb_df = df
        os.makedirs(os.path.dirname(csv), exist_ok=True)
    comb_df.to_csv(csv, index=False, header=False)

def create_log_map_array_ensamble(r_array, x_init, array_length, csv_path, every_xth_checkpoint=100):
    """
    Creates an ensemble array of logistic map traces and saves checkpoints to a CSV file.

    This function generates logistic map traces for given values of `r` in `r_array` and initial value `x_init`.
    The results are saved in a CSV file at specified intervals (`every_xth_checkpoint`). The function ensures that
    a valid path is provided for the CSV file and skips if no valid path can be created. Logistic map traces are computed
    using a helper function `create_log_map_trace`.

    If the process completes without filling a batch, the remaining entries in the batch are saved to the CSV as well.

    :param r_array: A numpy array containing the `r` values for which logistic map traces will be generated.
    :param x_init: The initial value `x` used for generating logistic map traces.
    :param array_length: The length of the logistic map time series to be calculated for each `r` value.
    :param csv_path: A string representing the file path where the checkpoints will be saved.
    :param every_xth_checkpoint: An integer specifying the interval at which checkpoints will be saved.
        Default value is 100.
    :return: Returns 0 if no valid path is provided for saving the CSV file, otherwise the function operates
        with no explicit return value.
    """
    csv_path = check_csv_path(csv_path)
    if not csv_path:
        print('No valid path was given or could be created, no ensamble will be created')
        return 0
    temp_arr = np.zeros((every_xth_checkpoint, array_length+1)) 
    i_local = 0
    for i, r in enumerate(r_array):
        i_local = i%every_xth_checkpoint
        if i_local==0 and i!=0:
            #save the checkpoint
            temp_df = pd.DataFrame(temp_arr)
            save_df_to_csv(temp_df, csv_path)
        temp_arr[i_local, 0] = r
        temp_arr[i_local, 1:] = create_log_map_trace(x_init, r, array_length)
    if i_local:
        temp_df = pd.DataFrame(temp_arr)
        save_df_to_csv(temp_df, csv_path)


def draw_logistic_map(x_arr):
    steps = np.arange(0,len(x_arr),1)
    plt.plot(steps, x_arr)
    plt.show(block=True)


if  __name__ == '__main__':
    install()
    # test_csv_path()
    # logtrace = create_log_map_trace(3, 3, 10)
    r_list = list(np.arange(0.5,4, 0.01))
    create_log_map_array_ensamble(r_list, 0.5, 10000, 'Logistic_maps/Long_Logistic_map.csv', 500)
