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

CSV processor for EEG analysis.

This module provides the CSVProcessor class for processing CSV data.
"""

import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly

from eeganalyzer.core.array_processor import Array_processor
from eeganalyzer.utils.buttler import Buttler


class CSVProcessor:
    """
    A class for processing CSV data.

    The CSVProcessor class facilitates the loading, preprocessing, and exporting of CSV data
    for further analysis. It provides multiple utilities, such as file loading, filtering,
    and calculating metrics.
    """

    def __init__(self, datapath: str, header=0, index=0, sfreq: int = None, remove_first_column: bool = False):
        self.datapath = datapath
        self.sfreq = sfreq
        self.remove_first_column = remove_first_column
        self.data = self.load_data_file(datapath, header, index)
        self.buttler = Buttler()  # Optional utility for handling file operations


    def load_data_file(self, data_file: str, header, index) -> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame and sets the sampling frequency.

        Args:
            data_file (str): Path to the CSV file to be loaded.
            sfreq (float, optional): Sampling frequency of the data. If None, it will be inferred from the data.

        Returns:
            tuple: (data, sfreq) where data is a pandas DataFrame and sfreq is the sampling frequency.
        """
        try:
            data = pd.read_csv(data_file, header=header, index_col=index)
            if data.empty:
                print(f"Warning: CSV file {data_file} is empty.")
                return None

            # If remove_time_vector is True, remove the first column
            if self.remove_first_column:
                data = data.iloc[:, 1:]
            return data
        except FileNotFoundError:
            print(f"File not found: {data_file}. Please check the filepath.")
        except pd.errors.EmptyDataError:
            print(f"File is empty: {data_file}.")
        except Exception as e:
            print(f"An error occurred while loading the file: {data_file}. Error: {e}")
        return None


    def downsample(self, resamp_freq):
        """
        Resamples the data to a new sampling frequency using scipy's resample_poly.

        Args:
        - resamp_freq (int): Target resampling frequency in Hz.

        Updates:
        - The `data` attribute is modified in place to reflect resampled data.
        """
        if resamp_freq is None or resamp_freq <= 0:
            print(f"Invalid resampling frequency: {resamp_freq}. Frequency must be a positive number.")
            return
        if self.sfreq and self.sfreq > resamp_freq:
            # Compute the integer factors for downsampling using resample_poly
            up = resamp_freq
            down = self.sfreq
            try:
                resampled_data = resample_poly(self.data.to_numpy(), up, down, axis=0)
                self.data = pd.DataFrame(resampled_data, columns=self.data.columns)  # Convert back to DataFrame
                self.sfreq = resamp_freq
            except Exception as e:
                print(f"An error occurred during resampling: {e}")
        else:
            print(f"Resampling frequency {resamp_freq} must be lower than the current sampling frequency {self.sfreq}.")


    def apply_filter(self, l_freq: float = None, h_freq: float = None, order: int = 5):
        """
        Applies zero-phase (two-pass) filtering to the data columns using filtfilt.
        Can perform high-pass, low-pass, or band-pass filtering.

        Args:
        - l_freq (float): The lower cutoff frequency for filtering (high-pass).
        - h_freq (float): The higher cutoff frequency for filtering (low-pass).
        - order (int): The order of the filter. Default is 5.

        Outputs:
        - None. The `data` attribute is modified in place.
        """
        if self.data is None or self.sfreq is None:
            print("Data not loaded or sampling frequency not set. Cannot apply filtering.")
            return
        try:
            # Calculate Nyquist frequency
            nyquist = 0.5 * self.sfreq

            # Set up filter parameters based on l_freq and h_freq
            if l_freq and h_freq:
                low = l_freq / nyquist
                high = h_freq / nyquist
                b, a = butter(order, [low, high], btype='band')
            elif l_freq:
                low = l_freq / nyquist
                b, a = butter(order, low, btype='high')
            elif h_freq:
                high = h_freq / nyquist
                b, a = butter(order, high, btype='low')
            else:
                print("No filtering performed as both l_freq and h_freq are not specified.")
                return

            # Apply zero-phase filtering with filtfilt
            numeric_data = self.data.to_numpy()
            filtered_data = filtfilt(b, a, numeric_data, axis=0)
            self.data = pd.DataFrame(filtered_data, columns=self.data.columns)

        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None, None

    def compute_metrics(self, metric_set_name: str, metric_path: str, outfile: str, l_freq=None, h_freq=None,
                        ep_start: int = None, ep_stop: int = None, ep_dur: int = None, overlap: int = 0,
                        resamp_freq=None, repeat_measurement: bool = False) -> str:
        """
        Compute metrics for CSV data.

        Args:
            metric_path (str): Path to the metric file.
            metric_set_name (str): Name of the metric set to calculate.
            outfile (str): File path where the resulting metrics (CSV) will be saved.
            l_freq (float, optional): Lower frequency cutoff. Defaults to None.
            h_freq (float, optional): Upper frequency cutoff. Defaults to None.
            ep_start (int, optional): Start offset for epoching in seconds. Defaults to None.
            ep_stop (int, optional): Stop offset for epoching in seconds. Defaults to None.
            ep_dur (int, optional): Duration of individual epochs in seconds. Defaults to None.
            overlap (int, optional): Amount of overlap between epochs in seconds. Defaults to 0.
            resamp_freq (float, optional): Frequency to which the data will be downsampled. Defaults to None.
            repeat_measurement (bool, optional): If True, recalculate metrics even if the output file exists. Defaults to False.

        Returns:
            str: A message indicating the outcome of the processing.
        """
        try:
            # Check the name of the outfile
            outfile_check, outfile_check_message = self.buttler.check_outfile_name(outfile, file_exists_ok=repeat_measurement)
            if not outfile_check:
                return outfile_check_message

            # Validate that data exists
            if self.data is None or self.sfreq is None:
                return "Data not loaded or sampling frequency not set."

            # Apply filtering
            self.apply_filter(l_freq=l_freq, h_freq=h_freq)

            # Downsample if required
            if resamp_freq:
                self.downsample(resamp_freq)

            # Initialize the ArrayProcessor for metric calculation
            array_processor = Array_processor(
                data=self.data,  # Processed data
                sfreq=self.sfreq,
                axis_of_time=0,
                metric_name=metric_set_name,
                metric_path=metric_path,
            )

            # Extract default or provided epoching parameters
            result_frame = array_processor.epoching(
                duration=ep_dur,
                start_time=ep_start,
                stop_time=ep_stop,
                overlap=overlap
            )

            # Save dataframe to csv
            if not result_frame.empty:
                result_frame.to_csv(outfile)
                return 'finished and saved successfully'
            else:
                return 'no metrics could be calculated'
        except Exception as e:
            return f'Error during metric computation: {str(e)}'