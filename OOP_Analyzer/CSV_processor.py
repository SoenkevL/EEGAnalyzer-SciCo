import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import Array_processor
import Buttler


class CSVProcessor:
    """
    A class for processing basic CSV files with support for sampling frequency
    adjustments, zero-phase filtering, and interaction with the ArrayProcessor class.
    
    Initialization:
    - `datapath` (str): Path to the CSV file to be processed.
    - `sfreq` (int): Sampling frequency of the data.
    - `time_vector` (bool): Whether the first column represents a time vector (default: False).
        If `True`, the first column will be excluded from processing.

    Methods:
    - `load_data_file(data_file)`: Loads a CSV file into a pandas DataFrame.
    - `downsample(resamp_freq)`: Resamples the data using scipy's resample_poly method.
    - `apply_filter(l_freq, h_freq)`: Applies two-pass zero-phase filtering using filtfilt.
    - `compute_metrics`: High-level function for metric calculation using ArrayProcessor.
    """

    def __init__(self, datapath: str, sfreq: int = None, time_vector: bool = False):
        self.datapath = datapath
        self.sfreq = sfreq
        self.time_vector = time_vector
        self.data = self.load_data_file(datapath)
        self.buttler = Buttler.Buttler()  # Optional utility for handling file operations

    def load_data_file(self, data_file: str):
        """
        Loads a CSV file into a pandas DataFrame.

        Args:
        - data_file (str): Path to the CSV file to be loaded.

        Returns:
        - data (pd.DataFrame): The loaded data as a DataFrame.
        """
        try:
            data = pd.read_csv(data_file)
            if data.empty:
                print(f"Warning: CSV file {data_file} is empty.")
                return None

            # If time_vector is True, remove the first column
            if self.time_vector:
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
            print(f"An error occurred during filtering: {e}")

    def compute_metrics(self, metric_set_name: str, outfile: str, ep_start: int = None, ep_stop: int = None,
                        ep_dur: int = None, overlap: int = 0, resamp_freq=None) -> str:
        """
        Compute metrics using ArrayProcessor. Saves the results to a specified output file.

        Args:
        - metric_set_name (str): Name of the metrics to compute.
        - outfile (str): Path of the output file where results will be saved.
        - ep_start (int, optional): Epoch start time (in seconds). Defaults to None.
        - ep_stop (int, optional): Epoch stop time (in seconds). Defaults to None.
        - ep_dur (int, optional): Duration of an epoch (in seconds). Defaults to None.
        - overlap (int, optional): Overlap between epochs (in seconds). Defaults to 0.
        - resamp_freq (int, optional): Sampling frequency for resampling.

        Returns:
        - str: Message indicating the processing status.
        """
        # Validate that data exists
        if self.data is None or self.sfreq is None:
            return "Data not loaded or sampling frequency not set."

        # Downsample if required
        if resamp_freq:
            self.downsample(resamp_freq)

        # Initialize the ArrayProcessor
        array_processor = Array_processor.Array_processor(
            data=self.data,  # Use the full data or whatever remains after earlier steps
            sfreq=self.sfreq,
            axis_of_time=0,
            metric_name=metric_set_name
        )

        # Compute metrics
        result_frame = array_processor.epoching(
            ep_dur=ep_dur, start_time=ep_start, stop_time=ep_stop, overlap=overlap
        )

        # Save results if metrics were successfully computed
        if not result_frame.empty:
            result_frame.to_csv(outfile, index=False)
            return "Metrics computed and saved successfully."
        else:
            return "No metrics could be calculated."