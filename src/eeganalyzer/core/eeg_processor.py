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

EEG processor for EEG analysis.

This module provides the EEG_processor class for processing EEG data.
"""
import os
from typing import Dict, List, Union

import mne
import numpy as np
import pandas as pd
import logging
from eeganalyzer.utils.buttler import check_outfile_name, find_task_from_filename
from custom_files import metrics, pipeline_preprocessing
from parallel_pandas import ParallelPandas
import time


#### general methods ####

def apply_metric_func(data: Union[np.ndarray, List[float]],
                      metric_name: str, **kwargs) -> Dict[str, float]:

    # Ensures EEG channel is saved as contiguous array in memory
    data = np.ascontiguousarray(data)
    try:
        result_dict = metrics.calculate(data, metric_name, **kwargs)
    except Exception as e:
        # Catch and log exceptions during default metric calculation
        logging.error(f"Could not apply metric"
                      f" to data with default parameters. Exception: {e}")
        return {}
    result_series = pd.Series(data=result_dict)
    return result_series

def calc_metrics_for_epoch(row, data_frame, sfreq, metric_name, **kwargs):
    start = row['start'] * sfreq
    stop = row['stop'] * sfreq
    current_epoch = data_frame.iloc[int(start):int(stop), 1:]
    if kwargs.get('channelwise', True): #by default we assume we want to calculate channelwise metrics
         result = current_epoch.apply(apply_metric_func, metric_name=metric_name, axis=0, **kwargs)#, executor='processes')
    else:
        #TODO: test code for multichannel application, so far not used
        result = apply_metric_func(current_epoch, **kwargs)
        result.name = 'all_channels'
    if isinstance(result, pd.Series):
        logging.warning('metric calculation returned a series not a dataframe. Something went wrong most probably, check results carefully')

    result['annot'] = row['annot']
    result['start'] = row['start']
    result['duration'] = row['stop']-row['start']
    result['metric'] = result.index
    result = result.reset_index(drop=True)
    result = result.loc[:, ['annot', 'start', 'duration', 'metric', *result.columns[:-4]]]
    return result

class EEG_processor:
    """
    A class for processing EEG (Electroencephalogram) data.

    The EEG_processor class facilitates the loading, preprocessing, and exporting of EEG data
    for further analysis. It provides multiple utilities, such as file loading, filtering,
    changing montages, downsampling, and calculating metrics.
    """
    def __init__(self, datapath, config, preload: bool = True):
        logging.debug(f'initialized EEGAnalyzer for datapath: {datapath} and config: {config} and preload: {preload}')
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.datapath = datapath
        self.config = config
        self.raw, _ = self._load_data_file(datapath, preload)
        logging.debug(f'initialized EEGAnalyzer for datapath: {datapath}')

        # initialize parallel-pandas
        num_system_cpus = os.cpu_count()
        logging.debug(f'number of system cpus: {num_system_cpus}')
        n_cpu = min(int(os.getenv('MAX_PROCESSORS')), num_system_cpus)
        logging.info(f'initializing parallel-pandas with {n_cpu} processes')
        ParallelPandas.initialize(disable_pr_bar=False, show_vmem=False, n_cpu=n_cpu)

    #### Loading and processing data ####
    def _load_data_file(self, data_file: str, preload: bool = True):
        """
        Loads an EEG file into an mne raw instance and extracts its sampling frequency.
        """
        valid_extensions = ['.fif', '.edf']
        if not any(data_file.endswith(ext) for ext in valid_extensions):
            logging.warning(f"Unsupported file type: {data_file}. Supported file types are: {', '.join(valid_extensions)}.")
            return None, None
        try:
            raw = mne.io.read_raw(data_file, preload=preload)
            sfreq = raw.info['sfreq']
            logging.debug(f"Loaded file: {data_file}")
            return raw, sfreq
        except FileNotFoundError:
            logging.error(f"File not found: {data_file}. Please check the filepath.")
        except ValueError:
            logging.error(f"Invalid file format: {data_file}. Unable to load data")
        except Exception:
            logging.error(f"An unexpected error occurred while loading the file: {data_file}")
        return None, None

    def _preprocess_eeg(self):
        self.raw = pipeline_preprocessing.preprocess_eeg(self.raw, self.config.get('preprocessing_name')
                                                         ,**self.config.get('preprocessing_params'))
        return self.raw

    #### Epoching ####
    def _epochs_from_annotation(self):
        start = self.config.get('start_time', 0)  # Default ep_start to 0 if None
        stop = self.config.get('stop_time', None)
        duration = self.config.get('duration', None)
        overlap = self.config.get('overlap', 0)
        relevant_annot_labels = self.config.get('annotations', [''])
        raw_annots = self.raw.annotations
        start_values = []
        stop_values = []
        annots = []
        if raw_annots:
            for annot in raw_annots:
                annot_name = annot['description']

                # Skip annotations not in relevant_annot_labels, if provided
                if relevant_annot_labels != 'all' and annot_name not in relevant_annot_labels:
                    continue

                # Extract start and duration of the annotation
                annot_start_seconds = annot['onset']
                annot_duration_seconds = annot['duration']
                annot_stop_seconds = annot_start_seconds + annot_duration_seconds

                logging.info(f'Processing annotation: {annot_name}, Times: {annot_start_seconds}-{annot_stop_seconds}')

                # Calculate epoch start and stop times
                start_time = annot_start_seconds + start
                stop_time = (min(start_time + stop, annot_stop_seconds)
                             if stop else annot_stop_seconds)

                t_start_times = np.array(
                    [t_start for t_start in np.arange(start_time, (stop_time - duration) + 1, duration - overlap)])
                t_stop_times = t_start_times + duration
                start_values.extend(t_start_times)
                stop_values.extend(t_stop_times)
                annots.extend([annot_name] * len(t_start_times))

        epochs = pd.DataFrame(columns=['annot', 'start', 'stop'])
        epochs['start'] = start_values
        epochs['stop'] = stop_values
        epochs['annot'] = annots
        return epochs

    def _epochs_from_whole_file(self):
        epochs = []
        start = self.config.get('start_time', 0)
        stop = self.config.get('stop_time', None)
        duration = self.config.get('ep_dur', None)
        task = find_task_from_filename(self.datapath)  # might be too specific due to prior use case
        if not duration:
            duration = int(stop - start)
        overlap = self.config.get('overlap', 0)
        # Compute metrics using the epoching function
        t_start_times = np.array(
            [t_start for t_start in np.arange(start, (stop - duration) + 1, duration - overlap)]
        )
        t_stop_times = t_start_times + duration

        epochs = pd.DataFrame(columns=['annot', 'start', 'stop'])
        epochs['start'] = t_start_times
        epochs['stop'] = t_stop_times
        epochs['annot'] = task
        return epochs

    ####################################################################################################################
    ############################################### public api #########################################################
    ####################################################################################################################

    def compute_metrics(self) -> str:
        # Check the name of the outfile
        t_start = time.time()
        outfile_check, outfile_check_message = check_outfile_name(self.config['outpath'], file_exists_ok=self.config['recompute'])
        if not outfile_check:
            return outfile_check_message

        # Process eeg
        self._preprocess_eeg()

        # Load data from EEG
        self.raw.load_data()
        data = self.raw.to_data_frame()
        logging.info(f'Data shape: {data.shape}')

        # Epoching
        relevant_annot_labels = self.config.get('annotations', None)
        if relevant_annot_labels:
            epochs = self._epochs_from_annotation()
        else:
            epochs = self._epochs_from_whole_file()

        # Initialize metrics
        logging.info(f'Calculating metrics:')
        # Compute metrics
        result_series = epochs.p_apply(calc_metrics_for_epoch, data_frame=data, sfreq=self.raw.info['sfreq'],
                                       metric_name=self.config.get('metric_name'), axis=1,
                                       executor='processes', **self.config.get('metric_params'))
        # Save dataframe to csv
        t_elapsed = time.time() - t_start
        if not result_series.empty:
            logging.info(f'Calculating metrics finished and took {t_elapsed:.2f} seconds')
            result_frame = pd.concat(result_series.to_list(), ignore_index=True, axis=0)
            # result_frame = result_series[0]
            # result_series[1:].apply(lambda x: pd.concat([result_frame, x], ignore_index=True, axis=0))
            result_frame = result_frame.sort_values(by=['annot', 'start'])
            result_frame = result_frame.reset_index(drop=True)
            # format the dataframe
            result_frame.to_csv(self.config['outpath'])
            logging.info(f'Results saved to {self.config["outpath"]}')
            return 'finished and saved successfully'
        else:
            logging.info(f'Calculating metrics failed and took {t_elapsed} seconds')
            return 'no metrics could be calculated'

