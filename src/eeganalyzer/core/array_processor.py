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
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import os, sys
import logging

class Array_processor:
    """
    This class provides a framework for processing array data, particularly for time-series analysis such as EEG data.
    It includes methods for setting attributes, calculating metrics, epoching data, and more.

    Attributes:
        data (pd.DataFrame): The input data (e.g., EEG data) to process.
        metric_name (str): The name of the metric or set of metrics to calculate.
        sfreq (float): The sampling frequency of the input data.
        axis_of_time (int): Axis indicating time (0 for rows, 1 for columns).
    """

    def __init__(self, data: Optional[pd.DataFrame] = None, metric_name: Optional[str] = None, metric_path: Optional[str] = None,
                 sfreq: Optional[float] = None, axis_of_time: int = 0, first_element_time=False):
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            self.data: Optional[pd.DataFrame] = None
            self.metric_name: Optional[str] = None
            self.metric_path: Optional[str] = None
            self.sfreq: Optional[float] = None
            self.axis_of_time: int = 0
            self.first_element_time = first_element_time
            self.time = None

            self.set_metric_name(metric_name)
            self.set_metric_path(metric_path)
            self.select_metrics = self.import_metrics()
            self.set_sfreq(sfreq)
            self.set_axis_of_time(axis_of_time)
            self.set_data(data)
            pass

    def import_metrics(self):
        """
        Dynamically imports the select_metrics function from a specified path.

        Returns:
            callable: The select_metrics function from the specified metrics file.

        Raises:
            ImportError: If the function cannot be imported from the specified path.
        """
        if not self.metric_path:
            logging.error("Metric path is not set. Use set_metric_path() first.")
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
                logging.error(f"Could not load spec for module at {self.metric_path}")
                raise ImportError(f"Could not load spec for module at {self.metric_path}")

            metrics_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metrics_module)

            # Get the select_metrics function
            if not hasattr(metrics_module, 'select_metrics'):
                logging.error(f"The metrics module at {self.metric_path} does not contain a select_metrics function")
                raise AttributeError(f"The metrics module at {self.metric_path} does not contain a select_metrics function")

            logging.debug(f"Successfully imported metrics from {self.metric_path}")
            return metrics_module.select_metrics

        except Exception as e:
            logging.error(f"Failed to import metrics from {self.metric_path}: {str(e)}")
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
            logging.error("Sampling frequency must be a positive number.")
            raise ValueError("Sampling frequency must be a positive number.")
        logging.debug(f"Setting sfreq to {sfreq}")
        self.sfreq = sfreq

    def set_data(self, data: pd.DataFrame):
        """
        Updates the data attribute.

        Parameters:
            data (pd.DataFrame): EEG data to process.
        """
        if self.first_element_time:
            if self.axis_of_time == 0:
                data = data.iloc[:, 1:]
                self.time = data.iloc[:, 0]

            else:
                data = data.iloc[1:, :]
                self.time = data.iloc[0, :]
        logging.debug(f"Setting data, first two rows:\n{data.head(2)}")
        self.data = data

    def set_axis_of_time(self, axis_of_time: int) -> None:
        """
        Sets the axis representing time in the data.
        
        Parameters:
            axis_of_time: Axis or dimension referring to time in the data.
        """
        if axis_of_time not in [1, 0]:
            logging.error("Axis of time must be either 1 (columns) or 0 (rows).")
            raise ValueError("Axis of time must be either 1 (columns) or 0 (rows).")
        logging.debug(f"Setting axis_of_time to {axis_of_time}")
        self.axis_of_time = axis_of_time

    def set_metric_name(self, metric_name: str) -> None:
        """
        Sets the name of the metric to calculate.

        Parameters:
            metric_name (str): Name of the metric.
        """
        if not isinstance(metric_name, str) or not metric_name.strip():
            logging.error("Metric name must be a non-empty string.")
            raise ValueError("Metric name must be a non-empty string.")
        logging.debug(f"Setting metric_name to {metric_name}")
        self.metric_name = metric_name

    def set_metric_path(self, metric_path: str) -> None:
        """
        Sets the name of the metric to calculate.

        Parameters:
            metric_path (str): Name of the metric.
        """
        if not isinstance(metric_path, str) or not metric_path.strip():
            logging.error("Metric path must be a non-empty string.")
            raise ValueError("Metric path must be a non-empty string.")
        if not os.path.exists(metric_path):
            logging.error("Metric path does not exist.")
            raise ValueError("Metric path does not exist.")
        logging.debug(f"Setting metric_path to {metric_path}")
        self.metric_path = metric_path
         
    def transpose_data(self) -> None:
        """
        Transposes the data based on the axis of time and updates the axis_of_time attribute.
        """
        if self.axis_of_time not in [0, 1]:
            logging.error("Axis of time must be either 0 (rows) or 1 (columns).")
            raise ValueError("Axis of time must be either 0 (rows) or 1 (columns).")

        if self.axis_of_time == 1:
            self.data = self.data.T
            self.axis_of_time = 0
        elif self.axis_of_time == 0:
            self.data = self.data.T
            self.axis_of_time = 1
        logging.debug(f"Transposed data, first two rows:\n{self.data.head(2)}")

    def initialize_metric_functions(self, name: str) -> Tuple[
        List[Callable[[Union[np.ndarray, List[float]]], Dict[str, float]]], bool]:
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
            logging.error("Metric set name must be a non-empty string.")
            raise ValueError("Metric set name must be a non-empty string.")

        metrics_functions = []
        channelwise = None
        try:
            metric_func_dict = self.select_metrics(name)
            metrics_functions = metric_func_dict.get('metric_funcs', None)
            channelwise = metric_func_dict.get('channelwise', True)

        except ValueError as ve:
            logging.error('metric functions need to return a dictionary containing:'
                          'metrics_functions, metrics_name_list, kwargs_list, channelwise')
        except Exception as e:
            logging.error("Error occurred while retrieving metrics for '%s': %s", name, str(e))
            raise ValueError(f"An error occurred while retrieving metrics for '{name}': {e}")

        return metrics_functions, channelwise

    @staticmethod
    def apply_metric_func(data: Union[np.ndarray, List[float]],
                          metric_func: Callable[[Union[np.ndarray, List[float]]], Dict[str, float]]) -> Dict[
        str, float]:
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
        logging.debug(f"Applying metric '{metric_func.__name__}' to data.")

        # Ensures EEG channel is saved as contiguous array in memory
        data = np.ascontiguousarray(data)

        try:
            return metric_func(data)
        except Exception as e:
            # Catch and log exceptions during default metric calculation
            logging.error(f"Could not apply metric '{metric_func.__name__}'"
                          f" to data with default parameters. Exception: {e}")
            return None

    def create_result_array(self, eeg_np_array, metrics_func_list: list) -> list:
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
        logging.debug(f"Creating result array for EEG data with shape {eeg_np_array.shape}.")
        return [self.apply_metric_func(eeg_np_array, metric_func)
                for metric_func in metrics_func_list]


    ############################################ advanced functions ########################################################

    def process_result_array(self, result_array: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, Any]], List[str]]:
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
        # Initialize processed array
        processed_array = []
        metric_name_list = []
        for result_dict in result_array:
            result_type = type(result_dict)
            try:
                if result_type == dict:
                    for key, value in result_dict.items():
                        processed_array.append((key, value))
                        metric_name_list.append(key)
            except Exception as e:
                logging.error(f"Error processing result: {result_dict}. Exception: {e}")
                processed_array.append(None)

        return processed_array, metric_name_list

    def create_result_dict_from_eeg_frame(self, data_frame: Union[pd.DataFrame, np.ndarray],
                                          metrics_func_list: List[callable],
                                          channelwise: bool) -> Tuple[Dict[Union[str, int], List[Tuple[str, Any]]], List[str]]:


        '''
        Creates a dictionary of computed metrics for EEG data.
        
        Parameters:
            data_frame (pd.DataFrame or np.ndarray): EEG data frame or numpy array where rows or columns represent time series.
            metrics_func_list (list): List of callable metric functions to be applied to the EEG data.
            channelwise (bool): If True, computes metrics for each time series individually; otherwise computes on the full data frame.
        
        Returns:
            tuple:
                - result_dict (dict): Dictionary with keys being EEG channels (columns/rows) and values as computed metrics.
                - metrics_name_list (list[str]): List of names for the computed metrics.
        
        Raises:
            ValueError: If metrics_func_list, metrics_name_list, or kwargs_list are not lists, or if their lengths do not match.
            TypeError: If data_frame is not a pd.DataFrame or np.ndarray.
        '''
        result_dict = {}
        metrics_name_list = []
        columns = data_frame.columns if isinstance(data_frame, pd.DataFrame) else range(data_frame.shape[1])
        data_frame = data_frame.to_numpy() if isinstance(data_frame, pd.DataFrame) else data_frame

        if channelwise:
            if self.axis_of_time == 0:
                for col, colname in zip(range(data_frame.shape[1]), columns):
                    try:
                        temp_data = data_frame[:, col]
                        raw_result_array = self.create_result_array(temp_data, metrics_func_list)
                        processed_result_array, metrics_name_list = self.process_result_array(raw_result_array)
                        result_dict[colname] = processed_result_array
                    except TypeError as te:
                        logging.error(f"Error processing column {colname} due to a type error. Please double check that all names for metric"
                                      f"processing are correct: \n {te} ")
                    except Exception as e:
                        logging.error(f"Error processing column {colname}: {e}")
                        result_dict[colname] = None
            else:
                for row in range(data_frame.shape[0]):
                    try:
                        temp_data = data_frame[row, :]
                        raw_result_array = self.create_result_array(temp_data, metrics_func_list)
                        processed_result_array, metrics_name_list = self.process_result_array(raw_result_array)
                        result_dict[row] = processed_result_array
                    except Exception as e:
                        logging.error(f"Error processing row {row}: {e}")
                        result_dict[row] = None
        else:
            try:
                raw_result_array = self.create_result_array(self.data, metrics_func_list)
                processed_result_array, metrics_name_list = self.process_result_array(raw_result_array)
                result_dict = {column: processed_result_array for column in range(self.data.shape[1])}
            except Exception as e:
                logging.error(f"Error processing entire data frame: {e}")
                result_dict = {}

        logging.debug(f"Result dictionary created: {result_dict}.")
        return result_dict, metrics_name_list

    @staticmethod
    def create_dataframe_from_result_dict(result_dict: Dict[Union[str, int], List[Tuple[str, Any]]], 
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
        index = pd.MultiIndex.from_product([[label], [start_data_record], [duration], metric_name_array],
                                           names=['label', 'startDataRecord', 'duration', 'metric'])
        eeg_column_names = list(result_dict.keys())

        # Initialize the DataFrame
        sub_results_frame = pd.DataFrame(columns=eeg_column_names,
                                         index=index,
                                         dtype=float)

        # Populate the DataFrame with metric results
        logging.debug(f"Populating DataFrame with results for metrics {metric_name_array}.")
        for column, result_array in result_dict.items():
            for result_tuple in result_array:
                if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                    logging.error("Each element in result_array must be a tuple of (metric_name, result).")
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
        try:
            # Initialize metrics to be calculated
            metrics_functions, channelwise = self.initialize_metric_functions(self.metric_name)

            # Calculate the results for the metrics and store them in a dictionary
            result_dict, metrics_name_list = self.create_result_dict_from_eeg_frame(
                dataframe, metrics_functions, channelwise
            )
            # Create the sub-results dataframe from the results dictionary
            sub_results_frame = self.create_dataframe_from_result_dict(
                result_dict, metrics_name_list, annot_startDataRecord, annot_duration, annot_label
            )
        except Exception as e:
            logging.error("Error occurred during calculating metrics for EEG dataframe.")
            raise RuntimeError("Error occurred during calculating metrics for EEG dataframe.") from e

        logging.debug(f"Sub-results dataframe created for EEG segment with label {annot_label}.")
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
            logging.warning("Duration must be a positive integer. Set to total duration.")

        # Validate and set stop_time
        if stop_time is None:
            stop_time = total_duration
        else:
            stop_time = min(total_duration, stop_time)  # Ensure stop_time is within the data range

        # Validate start_time
        if not start_time or start_time < 0 or start_time >= stop_time:
            start_time = 0
            logging.warning("Start time must be non-negative and less than stop time. set to 0")

        # Validate and adjust overlap
        if not overlap or overlap < 0 or overlap >= duration:
            logging.warning("Overlap not set or >= duration. Resetting overlap to 0.")
            overlap = 0

        # Check if duration fits within the interval [start_time, stop_time)
        if (stop_time - start_time) < duration:
            duration = stop_time - start_time
            logging.warning("The interval between start_time and stop_time is less than the duration. Setting duration to full interval.")

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
            logging.debug(f"Calculating metrics for epoch {t_onset} to {t_onset + duration} seconds.")
            sub_results_frame = self.calc_metrics_from_eeg_dataframe_and_annotations(
                eeg_dataframe, task, t_onset, duration
            )

            # Append results to the list
            results.append(sub_results_frame)

        # Combine all epochs into a single dataframe
        full_epoch_frame = pd.concat(results, axis=0) if results else pd.DataFrame()

        return full_epoch_frame