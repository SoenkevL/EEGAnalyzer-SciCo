import datetime as dt
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial
from rich.traceback import install


def Lorenz_step(input, dt, sigma, rho, beta):
    """
    Compute the derivatives of the Lorenz system at a given point in time.

    The Lorenz equations model atmospheric convection and are known for producing chaotic dynamics. This function calculates the rates of change (dx/dt, dy/dt, dz/dt) based on the current state [x, y, z] and given parameters.

    Parameters:
        input (array-like): A 3-element array or list containing the current values of x, y, and z.
        dt (float): The time step used in numerical integration.
        sigma (float): Prandtl number in the Lorenz equations.
        rho (float): Rayleigh number in the Lorenz equations.
        beta (float): Aspect ratio parameter in the Lorenz equations.

    Returns:
        numpy.ndarray: An array of three elements representing [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = input
    dx = (sigma * (y - x)) * dt
    dy = (x * (rho - z) - y) * dt
    dz = (x * y - beta * z) * dt
    return np.array([dx, dy, dz])


def init(input=[0, 0, 0], time=10, dt=0.001, sigma=1.0, rho=1.0, beta=1.0):
    return input, time, dt, sigma, rho, beta


def create_primitive_Lorenz_Attractor_trace(input, time, dt, sigma, rho, beta):
    state, time, dt, sigma, rho, beta = init(input, time, dt, sigma, rho, beta)
    N = int(time / dt)
    state_vec = np.zeros((N,3))
    state_vec[0] = state
    step = partial(Lorenz_step, dt=dt, sigma=sigma, rho=rho, beta=beta)
    for i in np.arange(1,N,1):
        state_vec[i] = state_vec[i-1]+step(input=state_vec[i-1])
    # print(state_vec)
    return state_vec


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


def save_df_to_csv(df: pd.DataFrame, csv: str, append: bool = True) -> None:
    # saves a dataframe to a csv path and tries to append if a csv at that path
    # already exists, otherwise creates a new csv
    if os.path.exists(csv):
        if append:
            temp_df = pd.read_csv(csv, header=0)
            comb_df = pd.concat([temp_df, df], ignore_index=True)  # Append rows, not columns
        else:
            comb_df = df
    else:
        comb_df = df
        os.makedirs(os.path.dirname(csv), exist_ok=True)
    comb_df.to_csv(csv, index=False, header=True)

def process_lorenz_output_array(input_array: np.ndarray) -> pd.DataFrame:
    '''
    This function processes the output of the Lorenz system. It goes from having an ndarray of form (constant_set, time, [x,y,z]) trace
    Important to notice is that in the last dimension of the input array, the very first row will always be the used constants, so instead of
    [x,y,z] it is [sigma, rho, beta]
    It will not save columns if their constants are only 0s
    The desired output is a pandas dataframe which has the following column scheme
    the first row (header) is [x-(sigma, rho, beta), y-(sigma, rho, beta), z-(sigma, rho, beta)] and the following rows the timevectors
    :param input_array: input array to process
    :return: the created dataframe
    '''
    input_shape = input_array.shape
    number_of_constantSets = input_array.shape[0]
    number_of_timepoints = input_array.shape[1]
    number_of_traces = input_array.shape[2]
    expanded_cols_length = int(number_of_constantSets * number_of_traces)
    expanded_rows_length = int(number_of_timepoints-1)
    new_array = []
    columns = []
    col_names = ['x', 'y', 'z']
    for const_set_idx in np.arange(number_of_constantSets):
        time_trace_array = input_array[const_set_idx]
        current_constants = time_trace_array[0, :]
        if any(current_constants):
            for trace in np.arange(number_of_traces):
                current_timevector = time_trace_array[1:, trace]
                column_name = f'{col_names[trace]}-{current_constants}'
                columns.append(column_name)
                new_array.append(current_timevector)
    if any(columns):
        new_array = np.array(new_array)
        new_array = np.transpose(new_array)
    return pd.DataFrame(new_array, columns=columns)



def create_primitive_Lorenz_attractor_array_ensamble(constants, state_init, time, dt,
                                            csv_path, every_xth_checkpoint=100):
    csv_path = check_csv_path(csv_path)
    N = int(time / dt)
    if not csv_path:
        print('No valid path was given or could be created, no ensamble will be created')
        return 0
    temp_arr = np.zeros((every_xth_checkpoint, N+1, 3))
    for i, const_vec in enumerate(constants):
        i_local = i%every_xth_checkpoint
        if i_local==0 and i!=0:
            #save the checkpoint
            temp_df = pd.DataFrame(temp_arr)
            save_df_to_csv(temp_df, csv_path)
        temp_arr[i_local, 0, :] = const_vec
        # I think I will have to do another loop here for assigning but lets try without first
        # Save to temp_arr using Lorenz attractor trace
        temp_arr[i_local, 1:] = create_primitive_Lorenz_Attractor_trace(
            state_init, time, dt, *const_vec
        )
    if np.any(temp_arr):
        # pandas only accepts 2d arrays as dataframes so we need to change the array first and create proper column labels
        temp_df = process_lorenz_output_array(temp_arr)
        save_df_to_csv(temp_df, csv_path)


def draw_logistic_map(x_arr):
    steps = np.arange(0,len(x_arr),1)
    plt.plot(steps, x_arr)
    plt.show(block=True)


if  __name__ == '__main__':
    install()
    # test_csv_path()
    # logtrace = create_log_map_trace(3, 3, 10)
    constants = [[10, 28, 8/3], [10.1, 28, 8/3],
                 [10.01, 28, 8/3], [10.001, 28, 8/3],
                 [5, 28, 8/3], [10, 28.1, 8/3],
                 [10, 20, 8/3], [10, 28, 8/4]]
    create_primitive_Lorenz_attractor_array_ensamble(constants, [1, 1, 1], 20, 0.001, 'Lorenz_maps/SmallEnsemble_Lorenz.csv', 100)
