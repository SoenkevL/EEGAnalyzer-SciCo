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

import mne
import pandas as pd
from icecream import ic
import logging

from eeganalyzer.core.array_processor import Array_processor
from eeganalyzer.utils.buttler import check_outfile_name, find_task_from_filename


class EEG_processor:
    """
    A class for processing EEG (Electroencephalogram) data.

    The EEG_processor class facilitates the loading, preprocessing, and exporting of EEG data
    for further analysis. It provides multiple utilities, such as file loading, filtering,
    changing montages, downsampling, and calculating metrics.
    """

    def __init__(self, datapath, config, preload: bool = True):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.datapath = datapath
        self.config = config
        self.raw, self.sfreq = self.load_data_file(datapath, preload)
        self.info = self.raw.info
        logging.debug(f'initialized EEGAnalyzer for datapath: {datapath}')

    #### Loading data ####
    def load_data_file(self, data_file: str, preload: bool = True):
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

    def load_data_of_raw_object(self):
        """
        Loads data from the raw EEG object into memory.
        """
        logging.debug("Loading data from the raw EEG object")
        self.raw.load_data()

    #### preprocessing data ####
    def downsample(self, resamp_freq=None):
        """
        Downsamples the EEG data to the specified sampling frequency.
        """
        resamp_freq = resamp_freq if resamp_freq else self.config['sfreq']
        if resamp_freq is None or resamp_freq <= 0:
            logging.warning(f"Invalid resampling frequency: {resamp_freq}. Frequency must be a positive number.")
            return
        if self.sfreq > resamp_freq:
            self.raw.resample(resamp_freq)
            self.sfreq = resamp_freq
        else:
            logging.warning(f"Resampling frequency {resamp_freq} must be lower than the current sampling frequency {self.sfreq}.")

    def apply_filter(self, l_freq: float = None, h_freq: float = None, picks: str = 'eeg'):
        """
        Filters a raw instance and returns it afterwards.
        """
        l_freq = l_freq if l_freq else self.config['l_freq']
        h_freq = h_freq if h_freq else self.config['h_freq']
        if l_freq and l_freq != 'None' and h_freq and h_freq != 'None':
            self.raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks)
            logging.debug(f"Filtered data from {l_freq} to {h_freq}")
        elif l_freq and l_freq != 'None':
            self.raw.filter(l_freq=l_freq, h_freq=self.raw.info['sfreq'], picks=picks)
            logging.debug(f"Filtered data from {l_freq}")
        elif h_freq and h_freq != 'None':
            self.raw.filter(l_freq=0, h_freq=h_freq, picks=picks)
            logging.debug(f"Filtered data to {h_freq}")
        else:
            logging.warning("No filtering performed as both l_freq and h_freq are not specified.")

    #### remontaging ####
    #Legacy code
    def ensure_electrodes_present(self, anodes, cathods, new_names):
        """
        checks if the anode and cathode for the bipolar reference are present, if not they will be dropped

        inputs:
        -anodes: a list of anode names which contains the name or None if the electrode is not present
        -cathodes: a list of cathode names which contains the name or None if the electrode is not present
        -new_names: a list of the bipolar channel names
        returns:
        -droped_names: all bipolar channels for which stuff was missing
        -new_names: new names which did not have to be dropped
        -anodes: anodes which have not been dropped
        -cathodes: cathodes which have not been dropped
        """
        drop_idx = []
        for i, (a, c, nn) in enumerate(zip(anodes, cathods, new_names)):
            if a and c:
                continue
            else:
                drop_idx.append(i)
        droped_names = [dn for i, dn in enumerate(new_names) if i in drop_idx]
        new_names = [nn for i, nn in enumerate(new_names) if i not in drop_idx]
        anodes = [a for i, a in enumerate(anodes) if i not in drop_idx]
        cathods = [c for i, c in enumerate(cathods) if i not in drop_idx]
        logging.debug(f'checked the present electrodes')
        logging.debug(f'The following channels could not be computed: {droped_names}')
        return anodes, cathods, new_names, droped_names

    #Legacy
    def only_keep_10_20_channels_and_check_bipolar(self):
        """
        Checks the EEG channels for containing a valid part, only once and no part that is marked as invalid.
        """
        valid_channel_parts = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                               'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
        invalid_channel_parts = ['pO2', 'CO2', 'X', 'Res', 'time']
        added_bads = []
        duplicate_positive = False
        for ch in self.raw.ch_names:
            bad_channel = True
            for vcp in valid_channel_parts:
                if vcp.lower() in ch.lower():
                    if bad_channel:
                        bad_channel = False
                    else:
                        duplicate_positive = True
            for icp in invalid_channel_parts:
                if icp.lower() in ch.lower():
                    bad_channel = True
            if bad_channel:
                added_bads.append(ch)
        self.raw.info['bads'] = self.raw.info['bads'] + added_bads
        self.raw.info['bads'] = list(set(self.raw.info['bads']))
        logging.debug(f'checked the 10-20 system channels')
        return duplicate_positive

    #Legacy
    def convert_electrode_names_to_channel_names(self, electrode_names: list[str], channel_names: list[str]):
        """
        Goes through the electrode names and converts them to channel names if the electrode name is part of the channel name

        inputs:
        -electrode_names: names of the electrodes from the montage
        -channel_names: names of the channels after conversion to ensure unified names in the outputs
        returns:
        -outputs_array: array of the same size as electrode_names containing either None or the according channel name
        """
        output_array = list()
        for i, e_name in enumerate(electrode_names):
            output_array.append(None)
            for c_name in channel_names:
                if e_name.lower() in c_name.lower():
                    output_array[i] = c_name
                    break
        logging.debug(f'converted the electrode names to channel names')
        return output_array

    def change_montage(self, montage: str = None):
        """
        Changes the montage of a raw instance. only works for 10-20 system channels
        """

        montage = montage if montage else self.config['montage']

        # Pick only EEG channels and exclude bad channels
        raw_internal = self.raw.pick(exclude='bads', picks='eeg').copy()
        
        # Apply the specified montage
        if montage == 'avg':
            ic(raw_internal.set_eeg_reference(ref_channels='average'))
            logging.debug(f'changed the montage to average')
        # change montage to doublebanana
        elif montage == 'doublebanana':
            anodes = ['Fp2', 'F8', 'T4', 'T6',
                      'Fp2', 'F4', 'C4', 'P4',
                      'Fz', 'Cz',
                      'Fp1', 'F3', 'C3', 'P3',
                      'Fp1', 'F7', 'T3', 'T5']
            cathodes = ['F8', 'T4', 'T6', 'O2',
                        'F4', 'C4', 'P4', 'O2',
                        'Cz', 'Pz',
                        'F3', 'C3', 'P3', 'O1',
                        'F7', 'T3', 'T5', 'O1']
            new_names = ['Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
                         'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                         'Fz-Cz', 'Cz-Pz',
                         'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                         'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1']
            # make sure names are unified and match the above
            anode_eeg_channels = self.convert_electrode_names_to_channel_names(anodes, raw_internal.ch_names)
            cathode_eeg_channels = self.convert_electrode_names_to_channel_names(cathodes, raw_internal.ch_names)
            try:
                logging.debug(f'changed the montage to doublebanana')
                # try setting the new bipolar reference
                ic(mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                             ch_name=new_names, copy=False))
            except ValueError:
                try:
                    # if setting failes in the previous step we need to check all channels are present and match
                    anode_eeg_channels, cathode_eeg_channels, new_names, dropped_names = self.ensure_electrodes_present(
                        anode_eeg_channels, cathode_eeg_channels, new_names)
                    logging.warning(
                        f'montage could not be set fully, probably not all needed channels are present. The following channels could not be computed: {dropped_names}')
                    mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                                 ch_name=new_names, copy=False)
                except ValueError:
                    logging.error('montage could not be set at all')
                    return None
        # set cirumferential montage
        elif montage == 'circumferential':
            anodes = ['Fp2', 'F8', 'T4', 'T6',
                      'O2', 'O1', 'T5', 'T3',
                      'F7', 'Fp1']
            cathodes = ['F8', 'T4', 'T6',
                        'O2', 'O1', 'T5', 'T3',
                        'F7', 'Fp1', 'Fp2']
            new_names = ['Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
                         'O2-O1', 'O1-T5', 'T5-T3', 'T3-F7',
                         'F7-Fp1', 'Fp1-Fp2']
            # unify names
            anode_eeg_channels = self.convert_electrode_names_to_channel_names(anodes, raw_internal.ch_names)
            cathode_eeg_channels = self.convert_electrode_names_to_channel_names(cathodes, raw_internal.ch_names)
            try:
                # set montage
                logging.debug(f'changed the montage to circumferential')
                mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                          ch_name=new_names, copy=False)
            except ValueError:
                # drop channels if montage could not be dropped
                anode_eeg_channels, cathode_eeg_channels, new_names, dropped_names = self.ensure_electrodes_present(
                    anode_eeg_channels, cathode_eeg_channels, new_names)
                logging.error(
                    f'montage could not be set fully, probably not all needed channels are present. The following channels could not be computed: {dropped_names}')
                mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                          ch_name=new_names, copy=False)
        # set montage to a specific channel
        elif montage in raw_internal.ch_names:
            raw_internal.set_eeg_reference(ref_channels=montage)
            logging.debug(f'changed the montage to {montage}')
        else:
            logging.warning(f'The given montage is not a viable option or a channel of the raw_internal object, no montage applied')
        
        return raw_internal

    def extract_eeg_columns(self):
        raw_copy = self.raw.copy()
        eeg_channels = raw_copy.pick(picks='eeg').ch_names
        logging.debug(f'extracted the eeg channels: {eeg_channels}')
        return eeg_channels

    #### metric calculation ####
    def calc_metric_from_annotations(self) -> pd.DataFrame:

        """
        Calculates metrics for EEG data based on annotations by segmenting them into epochs.

        Args:
        - metric_set_name (str): Name of the metric set to be applied.
        - ep_dur (int): Duration of each epoch in seconds.
        - ep_start (int): Start offset for each epoch relative to the annotation in seconds. Defaults to 0.
        - ep_stop (int): Stop offset or maximum duration of analyzed segments in seconds.
        - overlap (int, optional): Amount of overlap between epochs in seconds. Defaults to 0.
        - relevant_annot_labels (list of str, optional): List of annotation labels to analyze. If None, all annotations are used.

        Returns:
        - pandas.DataFrame: A dataframe containing metrics for all epochs segmented from the annotated EEG data.
            """
            # Load data from the raw EEG object and convert it to a DataFrame
        logging.debug(f'calculating metrics from annotations')
        self.raw.load_data()
        data = self.raw.to_data_frame()
        eeg_cols = self.extract_eeg_columns()
        logging.info(f'Data shape: {data.shape}')

        # Initialize the ArrayProcessor for metric calculations
        array_processor = Array_processor(
            data=data[eeg_cols],
            sfreq=self.sfreq,
            axis_of_time=0,
            metric_name=self.config['metric_set_name'],
            metric_path=self.config['metric_path'],
        )
        ep_start = self.config.get('start_time', 0)  # Default ep_start to 0 if None
        ep_stop = self.config.get('stop_time', None)
        ep_dur = self.config.get('duration', None)
        overlap = self.config.get('overlap', 0)
        relevant_annot_labels = self.config.get('annotations', [''])
        raw_annots = self.raw.annotations
        full_annot_frame = pd.DataFrame()
        sub_frame_list = []
        # Check if there are annotations in the EEG file
        if raw_annots:
            for annot in raw_annots:
                annot_name = annot['description']

                # Skip annotations not in relevant_annot_labels, if provided
                if relevant_annot_labels!='all' and annot_name not in relevant_annot_labels:
                    continue

                # Extract start and duration of the annotation
                annot_start_seconds = annot['onset']
                annot_duration_seconds = annot['duration']
                annot_stop_seconds = annot_start_seconds + annot_duration_seconds

                logging.info(f'Processing annotation: {annot_name}, Times: {annot_start_seconds}-{annot_stop_seconds}')

                # Calculate epoch start and stop times
                #TODO: naming here is shit
                ep_start_seconds = annot_start_seconds + ep_start
                ep_stop_seconds = (min(ep_start_seconds + ep_stop, annot_stop_seconds)
                                   if ep_stop else annot_stop_seconds)

                # Call the epoching function to calculate metrics
                logging.info(f'calculating metrics from annotation: '
                             f'{annot_name} from {ep_start_seconds} to {ep_stop_seconds} with duration {ep_dur} and overlap {overlap}')
                sub_results_frame = array_processor.epoching(
                    ep_dur, ep_start_seconds, ep_stop_seconds, overlap, annot_name
                )

                # Append metrics of the current annotation to the subframe list
                sub_frame_list.append(sub_results_frame)

        # create the final dataframe from all created subframes
        if len(sub_frame_list) > 0:
            full_annot_frame = pd.concat(sub_frame_list, axis=0)
        else:
            logging.warning('No annotations found in the EEG file.')

        return full_annot_frame

    def calc_metric_from_whole_file(self, task_label: str=None) -> pd.DataFrame:

        """
        Calculates metrics for the entire EEG file by segmenting it into epochs.

        Args:
        - metric_set_name (str): Name of the metric set to apply during computation.
        - ep_dur (int): Duration of each epoch in seconds.
        - ep_start (int): Start offset for each epoch relative to the beginning of the file, in seconds.
        - ep_stop (int): Stop offset or maximum duration of analyzed segments in seconds.
        - overlap (int, optional): Amount of overlap between epochs in seconds. Defaults to 0.
        - task_label (str, optional): Label for the task used in the epoching function. Defaults to None.

        Returns:
        - pandas.DataFrame: A dataframe containing the computed metrics for each channel across all epochs.
                            Each row corresponds to a specific segment of the EEG data.
        """
        # Load data from the raw EEG object into memory
        self.raw.load_data()

        # Convert raw data to a pandas DataFrame
        data = self.raw.to_data_frame()

        # Initialize the ArrayProcessor with relevant EEG data and parameters
        array_processor = Array_processor(
            data=data,
            sfreq=self.sfreq,
            axis_of_time=0,
            first_element_time=True,
            metric_name=self.config.get('metric_set_name', ''),
            metric_path=self.config.get('metric_path', '')
        )

        ep_start = self.config.get('start_time', 0)
        ep_stop = self.config.get('stop_time', None)
        ep_dur = self.config.get('ep_dur', None)
        if not ep_dur:
            ep_dur = int(ep_stop - ep_start)
        overlap = self.config.get('overlap', 0)
        # Compute metrics using the epoching function
        logging.info(f'calculating metrics from whole file with task label: '
                     f'{task_label} from {ep_start} to {ep_stop} with duration {ep_dur} and overlap {overlap}')
        result_frame = array_processor.epoching(
            ep_dur, ep_start, ep_stop, overlap, task_label
        )

        # Return the resulting DataFrame containing computed metrics
        return result_frame

    def compute_metrics_fif(self) -> pd.DataFrame:
        """
        Computes metrics for EEG data by handling files with or without annotations.

        This function acts as a wrapper around `calc_metric_from_annotations` and
        `calc_metric_from_whole_file`, allowing it to process EEG files with one,
        multiple, or no annotations.
        """

        # Check if annotation labels are provided
        relevant_annot_labels = self.config.get('annotations', None)
        if relevant_annot_labels:
                full_results_frame = self.calc_metric_from_annotations()
        else:
            task_label = find_task_from_filename(self.datapath) #might be too specific due to prior use case
            full_results_frame = self.calc_metric_from_whole_file(task_label)

        return full_results_frame


    ########################################################################################################################
    ######################################## high level functions ##########################################################
    ########################################################################################################################

    def compute_metrics(self) -> str:
        """
        Compute metrics for EEG processing

        using filtering, re-montaging, resampling, and metric calculations on individual EEG channels
        This is the primary function for processing and analyzing EEG data.

        Args:
        -dict: processing_config: A dictionary containing the processing parameters.
        Returns:
        - str: A message indicating the outcome of the processing. Possible messages:
               * 'finished and saved successfully': When computation and saving succeed.
               * 'no metrics could be calculated': When no metrics are computed.
               * Other messages depending on the checks performed in the function.

        Outputs:
        - Saves a CSV file with computed metrics at the specified `outfile` location.
        """
        try:
            # Check the name of the outfile
            outfile_check, outfile_check_message = check_outfile_name(self.config['outpath'], file_exists_ok=self.config['recompute'])
            if not outfile_check:
                return outfile_check_message

            # Only keeps channels which correspond to the typical 10-20 system names
            bipolar = self.only_keep_10_20_channels_and_check_bipolar()
            if bipolar:
                logging.info(f'Most likely already has a bipolar montage \nChannel names: \n {self.raw.ch_names}')

            # Filter
            self.apply_filter()

            # Downsample
            self.downsample()

            # Montage (also excludes bads and non-EEG channels even if no remontaging is done)
            #TODO: check if I really want this
            raw = self.change_montage()
            if not raw:
                return 'could not set montage, maybe EEG is faulty, skipping EEG'
            else:
                self.raw = raw

            # Calculate the metrics
            full_results_frame = self.compute_metrics_fif()

            # Save dataframe to csv
            if not full_results_frame.empty:
                full_results_frame.to_csv(self.config['outpath'])
                return 'finished and saved successfully'
            else:
                return 'no metrics could be calculated'
        except Exception as e:
            logging.error(f'Error during metric computation:')
            return f'Error during metric computation: {str(e)}'
