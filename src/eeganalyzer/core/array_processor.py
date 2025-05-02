"""
Copyright (C) <2025>  <Soenke van Loh>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

Array processor for EEG analysis.

This module provides the Array_processor class for processing array data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import os, sys

from eeganalyzer.utils.buttler import Buttler


class Array_processor:
    """
    This class provides a framework for processing array data, particularly for time-series analysis such as EEG data.
    It includes methods for setting attributes, calculating metrics, epoching data, and more.

    Attributes:
        data (pd.DataFrame): The input data (e.g., EEG data) to process.
        metric_name (str): The name of the metric or set of metrics to calculate.
        sfreq (float): The sampling frequency of the input data.
        axis_of_time (int): Axis indicating time (0 for rows, 1 for columns).
        buttler (Buttler): An object from the Buttler class to support auxiliary computations.

    Methods:
        set_sfreq(sfreq): Sets the sampling frequency.
        set_data(data): Updates the data attribute.
        set_axis_of_time(axis_of_time): Sets the axis representing time in the data.
        set_metric_name(metric_name): Sets the name of the metric to calculate.
        transpose_data(): Swaps rows and columns based on the axis of time.
        initialize_metric_functions(name): Loads metric functions, names, and arguments.
        apply_metric_func(data, metric_func, kwargs): Applies a metric function to a time-series.
        create_result_array(eeg_np_array, metrics_func_list, kwargs_list): Computes metrics for a given EEG data array.
        process_result_array(result_array, metric_name_array): Processes metric results for further use.
        create_result_dict_from_eeg_frame(data_frame, metrics_func_list, metrics_name_list, kwargs_list, channelwise=True):
            Computes metrics for EEG data and organizes results by channel or overall data.
        create_dataframe_from_result_dict(result_dict, metric_name_array, start_data_record, duration, label):
            Creates a DataFrame of computed metrics from a dictionary of results.
        calc_metrics_from_eeg_dataframe_and_annotations(dataframe, annot_label, annot_startDataRecord, annot_duration):
            Computes metrics for a designated EEG segment given its annotation details.
        epoching(duration, start_time=0, stop_time=None, overlap=0, task=None): 
            Divides data into epochs and calculates metrics for each, returning results in a DataFrame.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None, metric_name: Optional[str] = None, metric_path: Optional[str] = None,
                 sfreq: Optional[float] = None, axis_of_time: int = 0):
            self.data: Optional[pd.DataFrame] = None
            self.metric_name: Optional[str] = None
            self.metric_path: Optional[str] = None
            self.sfreq: Optional[float] = None
            self.axis_of_time: int = 0
            self.buttler: Buttler = Buttler()
            
            self.set_data(data)
            self.set_metric_name(metric_name)
            self.set_metric_path(metric_path)
            self.select_metrics = self.import_metrics()
            self.set_sfreq(sfreq)
            self.set_axis_of_time(axis_of_time)

    def import_metrics(self):
        """
        Dynamically imports the select_metrics function from a specified path.

        Returns:
            callable: The select_metrics function from the specified metrics file.

        Raises:
            ImportError: If the function cannot be imported from the specified path.
        """
        if not self.metric_path:
            raise ValueError("Metric path is not set. Use set_metric_path() first.")

        try:
            # Get the directory and filename
            dir_path = os.path.dirname(self.metric_path)
            file_name = os.path.basename(self.metric_path)

            # If it's a .py file, remove the extension
            if file_name.endswith('.py'):
                module_name = file_name[:-3]
            else:
                module_name = file_name

            # Add the directory to sys.path if it's not already there
            if dir_path not in sys.path:
                sys.path.insert(1, dir_path)

            # Dynamic import
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, self.metric_path)
            if not spec:
                raise ImportError(f"Could not load spec for module at {self.metric_path}")

            metrics_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metrics_module)

            # Get the select_metrics function
            if not hasattr(metrics_module, 'select_metrics'):
                raise AttributeError(f"The metrics module at {self.metric_path} does not contain a select_metrics function")

            return metrics_module.select_metrics

        except Exception as e:
            raise ImportError(f"Failed to import metrics from {self.metric_path}: {str(e)}")


    def set_sfreq(self, sfreq: float) -> None:
        """
        Sets the sampling frequency (sfreq) attribute.
    
        Parameters:
            sfreq (float): Sampling frequency of the data.
    
        Raises:
            ValueError: If sfreq is not a positive number.
        """
        if sfreq <= 0:
            raise ValueError("Sampling frequency must be a positive number.")
        self.sfreq = sfreq

    def set_data(self, data: pd.DataFrame):
        """
        Updates the data attribute.

        Parameters:
            data (pd.DataFrame): EEG data to process.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        self.data = data

    def set_axis_of_time(self, axis_of_time: int) -> None:
        """
        Sets the axis representing time in the data.
        
        Parameters:
            axis_of_time: Axis or dimension referring to time in the data.
        """
        if axis_of_time not in [1, 0]:
            raise ValueError("Axis of time must be either 1 (columns) or 0 (rows).")
        self.axis_of_time = axis_of_time

    def set_metric_name(self, metric_name: str) -> None:
        """
        Sets the name of the metric to calculate.

        Parameters:
            metric_name (str): Name of the metric.
        """
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("Metric name must be a non-empty string.")
        self.metric_name = metric_name

    def set_metric_path(self, metric_path: str) -> None:
        """
        Sets the name of the metric to calculate.

        Parameters:
            metric_path (str): Name of the metric.
        """
        if not isinstance(metric_path, str) or not metric_path.strip():
            raise ValueError("Metric path must be a non-empty string.")
        if not os.path.exists(metric_path):
            raise ValueError("Metric path does not exist.")
        self.metric_path = metric_path
         
    def transpose_data(self) -> None:
        """
        Transposes the data based on the axis of time and updates the axis_of_time attribute.
        """
        if self.axis_of_time not in [0, 1]:
            raise ValueError("Axis of time must be either 0 (rows) or 1 (columns).")

        if self.axis_of_time == 1:
            self.data = self.data.T
            self.axis_of_time = 0
        elif self.axis_of_time == 0:
            self.data = self.data.T
            self.axis_of_time = 1

    def initialize_metric_functions(self, name: str) -> Tuple[List[callable], List[str], List[Dict[str, Any]]]:
        """
        Loads the metric functions, their names, and corresponding arguments from the Metrics module.
        
        Parameters:
            name (str): Name of the metrics set to be loaded.
        
        Returns:
            tuple: A tuple containing:
                - metrics_functions (list): List of metric functions to calculate on the time series.
                - metrics_name_list (list): List of names for the functions, used to save the results.
                - kwargs_list (list): List of dictionaries with additional arguments for the functions.
        
        Raises:
            ValueError: If the name is not valid or no metrics are found for the given name.
            TypeError: If the output of Metrics.select_metrics is not a tuple of lists.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Metric set name must be a non-empty string.")

        try:
            metrics_functions, metrics_name_list, kwargs_list = self.select_metrics(name)

            if not isinstance(metrics_functions, list) or not isinstance(metrics_name_list, list) or not isinstance(
                    kwargs_list, list):
                raise TypeError("Output of Metrics.select_metrics must be three lists.")

            if not metrics_functions or not metrics_name_list or not kwargs_list:
                raise ValueError(f"No metrics found for the name: {name}")

        except Exception as e:
            raise ValueError(f"An error occurred while retrieving metrics for '{name}': {e}")

        return metrics_functions, metrics_name_list, kwargs_list

    def apply_metric_func(self, data: Union[np.ndarray, List[float]], 
                         metric_func: callable, 
                         kwargs: Optional[Dict[str, Any]]) -> Any:
        '''
        Applies a function to a timeseries (data channel).
        
        Inputs:
        - data: Channel data (one-dimensional time series).
        - metric_func: Function which is calculated based on the data.
        - kwargs: Additional arguments for the function.
        
        Returns:
        - Function output after calculation on the data.
        
        Raises:
        - ValueError if data is not one-dimensional.
        - TypeError if metric_func is not callable.
        '''
        # Ensure data is one-dimensional
        if not isinstance(data, (np.ndarray, list)) or len(np.shape(data)) != 1:
            raise ValueError("Data must be a one-dimensional time series.")

        # Ensure metric_func is callable
        if not callable(metric_func):
            raise TypeError("metric_func must be a callable function.")

        # Ensure kwargs is either None or a dictionary
        if kwargs is not None and not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dictionary or None.")

        # Ensures EEG channel is saved as contiguous array in memory
        data = np.ascontiguousarray(data)

        # Try applying the metric function with the given arguments
        if kwargs:
            try:
                # Try applying kwargs as keyword arguments
                return metric_func(data, **kwargs)
            except TypeError as e:
                # If kwargs are not accepted, use kwarg values as an arg list instead
                print(f"TypeError occurred: {e}. Retrying with positional arguments.")
                return metric_func(data, *list(kwargs.values()))
            except Exception as e:
                # Catch any unexpected exceptions and handle gracefully
                print(f"Could not apply metric '{metric_func.__name__}' to data. Exception: {e}")
                return None
        else:
            # If no kwargs are provided, calculate with default parameters
            try:
                return metric_func(data)
            except Exception as e:
                # Catch and log exceptions during default metric calculation
                print(
                    f"Could not apply metric '{metric_func.__name__}' to data with default parameters. Exception: {e}")
                return None

    def create_result_array(self, eeg_np_array, metrics_func_list: list, kwargs_list: list[dict]) -> list:
        '''
        Creates a list of computed metric results for the provided EEG data.
        
        Parameters:
        - eeg_np_array (np.ndarray): Numpy array containing EEG data, with each element representing a sample or channel.
        - metrics_func_list (list): List of callable metric functions to be applied to the EEG data.
        - kwargs_list (list[dict]): List of dictionaries containing additional arguments for each corresponding metric function.
        
        Returns:
        - list: A list of results where each result corresponds to the output of a metric function applied to the EEG data.
        
        Raises:
        - ValueError: If the input arguments are not structured as expected or contain invalid values.
        '''
        return [self.apply_metric_func(eeg_np_array, metric_func, kwargs)
                for metric_func, kwargs in zip(metrics_func_list, kwargs_list)]


    ############################################ advanced functions ########################################################

    def process_result_array(self, result_array: List[Any], metric_name_array: List[str]) -> List[Tuple[str, Any]]:
        '''
        Processes the results from calculated metrics and extracts relevant information for further use.
        
        Parameters:
        - result_array (list): List of metric results. Each result can be of type list, tuple, dict, or other supported formats.
        - metric_name_array (list[str]): List of metric names corresponding to each element in result_array.
        
        Returns:
        - processed_array (list): List of tuples where each tuple contains:
            (metric_name, extracted_value).
        
        Raises:
        - ValueError: If metric_name_array and result_array lengths do not match.
        '''
        if not isinstance(result_array, list):
            raise TypeError("result_array must be a list.")
        if not isinstance(metric_name_array, list):
            raise TypeError("metric_name_array must be a list.")
        if len(result_array) != len(metric_name_array):
            raise ValueError("result_array and metric_name_array must have the same length.")

        # Initialize processed array
        processed_array = []
        for result in result_array:
            result_type = type(result)
            try:
                if result_type in (list, tuple):
                    if len(result) > 0:
                        processed_array.append(result[0])
                    else:
                        processed_array.append(None)
                elif result_type == dict:
                    for key, value in result.items():
                        if key == 'result':
                            value = self.buttler.map_chaos_pipe_result_to_float(value)
                        processed_array.append(value)
                else:
                    processed_array.append(result)  # Handle other result types directly
            except Exception as e:
                print(f"Error processing result: {result}. Exception: {e}")
                processed_array.append(None)

        # Create tuples with names
        try:
            for i, (name, value) in enumerate(zip(metric_name_array, processed_array)):
                processed_array[i] = (name, value)
        except Exception as e:
            raise RuntimeError("Error occurred while pairing metric names with results.") from e

        return processed_array

    def create_result_dict_from_eeg_frame(self, data_frame: Union[pd.DataFrame, np.ndarray], 
                                          metrics_func_list: List[callable],
                                          metrics_name_list: List[str], 
                                          kwargs_list: List[Dict[str, Any]],
                                          channelwise: bool = True) -> Tuple[Dict[Union[str, int], List[Tuple[str, Any]]], List[str]]:

        '''
        Creates a dictionary of computed metrics for EEG data.
        
        Parameters:
            data_frame (pd.DataFrame or np.ndarray): EEG data frame or numpy array where rows or columns represent time series.
            metrics_func_list (list): List of callable metric functions to be applied to the EEG data.
            metrics_name_list (list[str]): List of names corresponding to the metric functions.
            kwargs_list (list[dict]): List of dictionaries containing additional arguments for the metric functions.
            channelwise (bool): If True, computes metrics for each time series individually; otherwise computes on the full data frame.
        
        Returns:
            tuple:
                - result_dict (dict): Dictionary with keys being EEG channels (columns/rows) and values as computed metrics.
                - metrics_name_list (list[str]): List of names for the computed metrics.
        
        Raises:
            ValueError: If metrics_func_list, metrics_name_list, or kwargs_list are not lists, or if their lengths do not match.
            TypeError: If data_frame is not a pd.DataFrame or np.ndarray.
        '''
        if not isinstance(metrics_func_list, list) or not isinstance(metrics_name_list, list) or not isinstance(kwargs_list,
                                                                                                                list):
            raise ValueError("metrics_func_list, metrics_name_list, and kwargs_list must all be lists.")
        if len(metrics_func_list) != len(metrics_name_list) or len(metrics_func_list) != len(kwargs_list):
            raise ValueError("metrics_func_list, metrics_name_list, and kwargs_list must have the same length.")
        if not isinstance(data_frame, (pd.DataFrame, np.ndarray)):
            raise TypeError("data_frame must be a pandas DataFrame or a numpy array.")

        result_dict = {}
        columns = data_frame.columns if isinstance(data_frame, pd.DataFrame) else range(data_frame.shape[1])
        data_frame = data_frame.to_numpy() if isinstance(data_frame, pd.DataFrame) else data_frame

        if channelwise:
            if self.axis_of_time == 0:
                for col, colname in zip(range(data_frame.shape[1]), columns):
                    try:
                        temp_data = data_frame[:, col]
                        raw_result_array = self.create_result_array(temp_data, metrics_func_list, kwargs_list)
                        processed_result_array = self.process_result_array(raw_result_array, metrics_name_list)
                        result_dict[colname] = processed_result_array
                    except Exception as e:
                        print(f"Error processing column {colname}: {e}")
                        result_dict[colname] = None
            else:
                for row in range(data_frame.shape[0]):
                    try:
                        temp_data = data_frame[row, :]
                        raw_result_array = self.create_result_array(temp_data, metrics_func_list, kwargs_list)
                        processed_result_array = self.process_result_array(raw_result_array, metrics_name_list)
                        result_dict[row] = processed_result_array
                    except Exception as e:
                        print(f"Error processing row {row}: {e}")
                        result_dict[row] = None
        else:
            try:
                raw_result_array = self.create_result_array(self.data, metrics_func_list, kwargs_list)
                processed_result_array = self.process_result_array(raw_result_array, metrics_name_list)
                result_dict = {column: processed_result_array for column in range(self.data.shape[1])}
            except Exception as e:
                print(f"Error processing entire data frame: {e}")
                result_dict = {}

        return result_dict, metrics_name_list

    def create_dataframe_from_result_dict(self, result_dict: Dict[Union[str, int], List[Tuple[str, Any]]], 
                                          metric_name_array: List[str],
                                          start_data_record: float, 
                                          duration: float, 
                                          label: str = '<missing>') -> pd.DataFrame:

        '''
        Generates a DataFrame using the given result dictionary and annotation details.

        Parameters:
            result_dict (dict): Dictionary where keys are EEG channel names and values are lists of metric results.
            metric_name_array (list[str]): List of metric names corresponding to the computed results.
            start_data_record (float): Start time of the EEG segment in seconds.
            duration (float): Duration of the EEG segment in seconds.
            label (str): Label associated with the EEG segment.

        Returns:
            pd.DataFrame: A DataFrame containing the metrics per channel with multi-indexing on label, startDataRecord,
                          duration, and metric.

        Raises:
            ValueError: If the result_dict or metric_name_array are invalid.
            TypeError: If input arguments are not of expected types.
        '''
        if not isinstance(result_dict, dict):
            raise TypeError("result_dict must be a dictionary.")
        if not isinstance(metric_name_array, list) or not all(isinstance(item, str) for item in metric_name_array):
            raise TypeError("metric_name_array must be a list of strings.")
        if not isinstance(start_data_record, (int, float)) or start_data_record < 0:
            raise ValueError("start_data_record must be a non-negative number.")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError("duration must be a positive number.")
        if not isinstance(label, (str, int, float)):
            label = '<missing>'
            print('no label was provided, using <missing> instead.')
        # Create multi-index based on metric, label, startDataRecord, and duration
        index = pd.MultiIndex.from_product([[label], [start_data_record], [duration], metric_name_array],
                                           names=['label', 'startDataRecord', 'duration', 'metric'])
        eeg_column_names = list(result_dict.keys())

        # Initialize the DataFrame
        sub_results_frame = pd.DataFrame(columns=eeg_column_names,
                                         index=index,
                                         dtype=float)

        # Populate the DataFrame with metric results
        for column, result_array in result_dict.items():
            if not isinstance(result_array, list):
                raise TypeError(f"Values in result_dict must be lists, but got {type(result_array)} for column '{column}'.")
            for result_tuple in result_array:
                if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                    raise ValueError("Each element in result_array must be a tuple of (metric_name, result).")
                metric_name, result = result_tuple
                if metric_name in metric_name_array:
                    sub_results_frame.loc[(label, start_data_record, duration, metric_name), column] = result

        return sub_results_frame

    def calc_metrics_from_eeg_dataframe_and_annotations(self, dataframe: pd.DataFrame,
                                                        annot_label: Union[str, int, float], 
                                                        annot_startDataRecord: float,
                                                        annot_duration: float) -> pd.DataFrame:

        '''
        Combines steps to compute metrics, process results, and create a sub-dataframe for the EEG segment.

        Parameters:
            dataframe (pd.DataFrame): EEG data to be analyzed, where rows or columns represent time series data.
            annot_label (str): Label of the annotation associated with the EEG segment.
            annot_startDataRecord (float): Start time of the annotated segment in seconds.
            annot_duration (float): Duration of the annotated segment in seconds.

        Returns:
            pd.DataFrame: Dataframe containing computed metrics per channel for the segment.

        Raises:
            ValueError: If metrics cannot be initialized, or processing any step fails.
            TypeError: If inputs are not of the expected type.
        '''
        # Validate inputs
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if not isinstance(annot_label, (str, int, float)):
            annot_label = '<missing>'
            print('no label was provided, using <missing> instead.')


        try:
            # Initialize metrics to be calculated
            metrics_functions, metrics_name_list, kwargs_list = self.initialize_metric_functions(self.metric_name)

            # Calculate the results for the metrics and store them in a dictionary
            result_dict, metrics_name_list = self.create_result_dict_from_eeg_frame(
                dataframe, metrics_functions, metrics_name_list, kwargs_list
            )

            # Create the sub-results dataframe from the results dictionary
            sub_results_frame = self.create_dataframe_from_result_dict(
                result_dict, metrics_name_list, annot_startDataRecord, annot_duration, annot_label
            )
        except Exception as e:
            raise RuntimeError("Error occurred during calculating metrics for EEG dataframe.") from e

        return sub_results_frame

    def epoching(self, duration: int, start_time: int = 0, stop_time: Optional[int] = None,
                 overlap: int = 0, task: Optional[str] = None) -> pd.DataFrame:
        """
        Divide data into epochs and calculate metrics for each epoch.

        Parameters:
            duration (int): Length of each epoch in seconds (mandatory).
            start_time (int): Start time in seconds for epoching. Defaults to 0.
            stop_time (int): End time in seconds for epoching. Defaults to total duration.
            overlap (int): Overlap in seconds between consecutive epochs. Defaults to 0.
            task (str): Task label for metrics calculation (optional).

        Returns:
            pd.DataFrame: A dataframe containing calculated metrics for all epochs.
        """

        # Determine the total duration (in seconds) based on the data length and sampling frequency
        total_duration = np.round(len(self.data) / self.sfreq)
        # Validate duration
        if not duration or duration <= 0:
            duration = total_duration
            print("Duration must be a positive integer. Set to total duration.")

        # Validate and set stop_time
        if stop_time is None:
            stop_time = total_duration
        else:
            stop_time = min(total_duration, stop_time)  # Ensure stop_time is within the data range

        # Validate start_time
        if not start_time or start_time < 0 or start_time >= stop_time:
            start_time = 0
            print("Start time must be non-negative and less than stop time. set to 0")

        # Validate and adjust overlap
        if not overlap or overlap < 0 or overlap >= duration:
            print("Overlap not set or >= duration. Resetting overlap to 0.")
            overlap = 0

        # Check if duration fits within the interval [start_time, stop_time)
        if (stop_time - start_time) < duration:
            duration = stop_time - start_time
            print("The interval between start_time and stop_time is less than the duration. Setting duration to full interval.")

        # Initialize results container
        results = []

        # Iterate through epochs
        for t_onset in np.arange(start_time, (stop_time - duration) + 1, duration - overlap):
            t_onset = int(t_onset)
            t_onset_samples = int(t_onset * self.sfreq)  # Convert time to sample index
            t_stop_samples = int((t_onset + duration) * self.sfreq)  # Calculate end sample index

            # Extract the EEG dataframe for this epoch
            eeg_dataframe = self.data.iloc[t_onset_samples:t_stop_samples, :]

            # Calculate metrics for the current epoch
            print(f'Calculating for times: {t_onset} to {t_onset + duration} seconds')
            sub_results_frame = self.calc_metrics_from_eeg_dataframe_and_annotations(
                eeg_dataframe, task, t_onset, duration
            )

            # Append results to the list
            results.append(sub_results_frame)

        # Combine all epochs into a single dataframe
        full_epoch_frame = pd.concat(results, axis=0) if results else pd.DataFrame()

        return full_epoch_frame