import mne
import pandas as pd
from icecream import ic

import Array_processor
import Buttler


class EEG_processor:
    """
    A class for processing EEG (Electroencephalogram) data.

    The EEG_processor class facilitates the loading, preprocessing, and exporting of EEG data
    for further analysis. It provides multiple utilities, such as file loading, filtering,
    changing montages, downsampling, and calculating metrics.

    Initialization:
    - `datapath` (str): Path to the EEG file to be processed.
    - 'preload' (bool): If the data of the eeg should be loaded immediately. Default is True.

    Methods:
    - `load_data_file(data_file, preload)`: Loads a file into an `mne` raw object and extracts its sampling frequency.
                                            If reload is True (default), the data is loaded immediately.
    - `load_data_of_raw_object()`: Loads the actual data into memory from the raw object.
    - `downsample(resamp_freq)`: Downsamples the EEG data to a specified frequency.
    - `apply_filter(l_freq, h_freq, picks)`: Applies high-pass, low-pass, band-pass, or band-stop filters.
    - `ensure_electrodes_present(anodes, cathods, new_names)`: Ensures electrodes for bipolar referencing are present.
    - `only_keep_10_20_channels_and_check_bipolar()`: Filters EEG channels to keep only those adhering to the 10-20 system.
    - `convert_electrode_names_to_channel_names(electrode_names, channel_names)`: Maps electrode names to channel names.
    - `change_montage(montage)`: Changes the montage of the EEG data.
    - `extract_eeg_columns(eeg_dataframe)`: Extracts channel names (excluding the time column) from an EEG dataframe.
    - `calc_metric_from_annotations(name, ep_dur, ep_start, ep_stop, overlap, relevant_annot_labels)`: Calculates metrics for annotated segments.
    - `calc_metric_from_whole_file(name, ep_dur, ep_start, ep_stop, overlap, task_label)`: Calculates metrics for the entire EEG file.
    - `compute_metrics_fif(name, relevant_annot_labels, ep_dur, ep_start, ep_stop, overlap, task_label)`: Wraps annotation-based and full-file metric calculations.
    - `compute_metrics(name, annot, outfile, lfreq, hfreq, montage, ep_dur, ep_start, ep_stop, overlap, resamp_freq)`: Combines preprocessing and metric calculations for all or specific annotations and saves results to a file.
    """

    def __init__(self, datapath, preload: bool = True):
        self.datapath = datapath
        self.raw, self.sfreq = self.load_data_file(datapath, preload)
        self.info = self.raw.info
        self.buttler = Buttler.Buttler()

    def load_data_file(self, data_file: str, preload: bool = True):
        """
        Loads an EEG file into an mne raw instance and extracts its sampling frequency.

        Performs basic file type validation to ensure compatibility.

        Inputs:
        - data_file (str): Path to the file to be loaded. Supports `.fif` and `.edf` file formats.

        Outputs:
        - raw: mne raw instance containing the EEG data.
        - sfreq: Sampling frequency of the EEG data.
        """
        valid_extensions = ['.fif', '.edf']
        if not any(data_file.endswith(ext) for ext in valid_extensions):
            print(f"Unsupported file type: {data_file}. Supported file types are: {', '.join(valid_extensions)}.")
            return None, None
        try:
            raw = mne.io.read_raw(data_file, preload=preload)
            sfreq = raw.info['sfreq']
            return raw, sfreq
        except FileNotFoundError:
            print(f"File not found: {data_file}. Please check the filepath.")
        except ValueError as ve:
            print(f"Invalid file format: {data_file}. Unable to load data. Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred while loading the file: {data_file}. Error: {e}")
        return None, None

    def load_data_of_raw_object(self):
        """
        Loads data from the raw EEG object into memory.

        This ensures that the raw object is fully loaded
        and ready for downstream processing.
        """
        self.raw.load_data()

    def downsample(self, resamp_freq):
        """
        Downsamples the EEG data to the specified sampling frequency.

        Inputs:
        - resamp_freq (int): Target resampling frequency. Must be less than the current frequency.

        Outputs:
        - None. Modifies the raw EEG object in place.
        """
        if resamp_freq is None or resamp_freq <= 0:
            print(f"Invalid resampling frequency: {resamp_freq}. Frequency must be a positive number.")
            return
        if self.sfreq > resamp_freq:
            self.raw.resample(resamp_freq)
            self.sfreq = resamp_freq
        else:
            print(f"Resampling frequency {resamp_freq} must be lower than the current sampling frequency {self.sfreq}.")

    def apply_filter(self, l_freq: float = None, h_freq: float = None, picks: str = 'eeg'):
        """
        filters a raw instance and returns it afterwards, can be used for lowpass, highpass, bandpass, bandstop filter
        inputs:
        -raw: raw instance (required)
        -l_freq: the lower end of frequencies that should be passed, so essentially only frequencies above it will be included
        -h_freq: the higher end of frequencies that should be passed, so essentially only frequencies below it will be included
        -picks: makes sure only eeg channels are used which can be helpful from time to time
        outputs:
        -None, instead alters the inputed raw instance
        """
        if l_freq and l_freq != 'None' and h_freq and h_freq != 'None':
            self.raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks)
        elif l_freq and l_freq != 'None':
            self.raw.filter(l_freq=l_freq, h_freq=self.raw.info['sfreq'], picks=picks)
        elif h_freq and h_freq != 'None':
            self.raw.filter(l_freq=0, h_freq=h_freq, picks=picks)
        else:
            print("No filtering performed as both l_freq and h_freq are not specified.")

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
        return anodes, cathods, new_names, droped_names

    def only_keep_10_20_channels_and_check_bipolar(self):
        """
        Checks the EEG channels for containing a valid part, only once and no part that is marked as invalid.

        In order to only keep the correct EEG channels and marke others as bad
        returns True if two valid channel parts are in one channel
        inputs:
        -raw: raw instance from EEG
        outputs:
        -duplicate_positive: returns True if a bipolar montage is used as the channel names contain two channels
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
        return duplicate_positive

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
        return output_array

    def change_montage(self, montage: str):
        """
        changes the montage of a raw instance

        input:
        -raw: raw instance of the EEG
        -montage: name of the montage, either 'avg', 'doublebanana', 'circumferential' or a specific channel
        returns:
        -raw_internal: the changed raw instance, can be None if there is an error during computation
        """
        # in this step now automatically some channels are excluded if marked as bad or not marked as EEG channels
        raw_internal = self.raw.pick(exclude='bads', picks='eeg').copy()
        # change the reference used in the eeg file to avg
        if montage == 'avg':
            ic(raw_internal.set_eeg_reference(ref_channels='average'))
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
                # try setting the new bipolar reference
                ic(mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                             ch_name=new_names, copy=False))
            except ValueError:
                try:
                    # if setting failes in the previous step we need to check all channels are present and match
                    anode_eeg_channels, cathode_eeg_channels, new_names, dropped_names = self.ensure_electrodes_present(
                        anode_eeg_channels, cathode_eeg_channels, new_names)
                    print(
                        f'montage could not be set fully, probably not all needed channels are present. The following channels could not be computed: {dropped_names}')
                    ic(mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                                 ch_name=new_names, copy=False))
                except ValueError:
                    print('montage could not be set at all')
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
                mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                          ch_name=new_names, copy=False)
            except ValueError:
                # drop channels if montage could not be dropped
                anode_eeg_channels, cathode_eeg_channels, new_names, dropped_names = self.ensure_electrodes_present(
                    anode_eeg_channels, cathode_eeg_channels, new_names)
                print(
                    f'montage could not be set fully, probably not all needed channels are present. The following channels could not be computed: {dropped_names}')
                mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                          ch_name=new_names, copy=False)
        # set montage to a specific channel
        elif montage in raw_internal.ch_names:
            raw_internal.set_eeg_reference(ref_channels=montage)
        else:
            print(f'The given montage is not a viable option or a channel of the raw_internal object, no montage applied')
        return raw_internal

    def extract_eeg_columns(self, eeg_dataframe):
        """
        Extracts all besides the first column from the eeg dataframe

        inputs:
        -dataframe with eeg data
        returns:
        -list of channels
        """
        return eeg_dataframe.columns[1:]

    def calc_metric_from_annotations(self, metric_set_name, ep_dur: int, ep_start: int, ep_stop: int,
                                     overlap: int = 0, relevant_annot_labels: list = None) -> pd.DataFrame:

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
        self.raw.load_data()
        data = self.raw.to_data_frame()
        eeg_cols = self.extract_eeg_columns(data)
        print(f'Data shape: {data.shape}')

        # Initialize the ArrayProcessor for metric calculations
        array_processor = Array_processor.Array_processor(
            data=data[eeg_cols],
            sfreq=self.sfreq,
            axis_of_time=0,
            metric_name=metric_set_name
        )
        ep_start = ep_start or 0  # Default ep_start to 0 if None
        raw_annots = self.raw.annotations
        full_annot_frame = pd.DataFrame()
        sub_frame_list = []
        # Check if there are annotations in the EEG file
        if raw_annots:
            for annot in raw_annots:
                annot_name = annot['description']

                # Skip annotations not in relevant_annot_labels, if provided
                if relevant_annot_labels and annot_name not in relevant_annot_labels:
                    continue

                # Extract start and duration of the annotation
                annot_start_seconds = annot['onset']
                annot_duration_seconds = annot['duration']
                annot_stop_seconds = annot_start_seconds + annot_duration_seconds

                print(f'Processing annotation: {annot_name}, Times: {annot_start_seconds}-{annot_stop_seconds}')

                # Calculate epoch start and stop times
                ep_start_seconds = annot_start_seconds + ep_start
                ep_stop_seconds = (min(ep_start_seconds + ep_stop, annot_stop_seconds)
                                   if ep_stop else annot_stop_seconds)

                # Call the epoching function to calculate metrics
                sub_results_frame = array_processor.epoching(
                    ep_dur, ep_start_seconds, ep_stop_seconds, overlap, annot_name
                )

                # Append metrics of the current annotation to the subframe list
                sub_frame_list.append(sub_results_frame)

        # create the final dataframe from all created subframes
        if len(sub_frame_list) > 0:
            full_annot_frame = pd.concat(sub_frame_list, axis=0)
        else:
            print('No annotations found in the EEG file.')

        return full_annot_frame

    def calc_metric_from_whole_file(self, metric_set_name, ep_dur: int, ep_start: int, ep_stop: int,
                                    overlap: int = 0, task_label: str = None) -> pd.DataFrame:

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

        # Extract EEG channel columns (excluding the time column)
        eeg_cols = self.extract_eeg_columns(data)

        # Initialize the ArrayProcessor with relevant EEG data and parameters
        array_processor = Array_processor.Array_processor(
            data=data.loc[:,eeg_cols],
            sfreq=self.sfreq,
            axis_of_time=0,
            metric_name=metric_set_name
        )

        # Compute metrics using the epoching function
        result_frame = array_processor.epoching(
            ep_dur, ep_start, ep_stop, overlap, task_label
        )

        # Return the resulting DataFrame containing computed metrics
        return result_frame

    def compute_metrics_fif(self, metric_name, relevant_annot_labels: list = None,
                            ep_dur=None, ep_start=None, ep_stop=None, overlap: int = 0,
                            task_label=None) -> pd.DataFrame:

        """
        Computes metrics for EEG data by handling files with or without annotations.

        This function acts as a wrapper around `calc_metric_from_annotations` and
        `calc_metric_from_whole_file`, allowing it to process EEG files with one,
        multiple, or no annotations.

        Args:
        - metric_name (str): Name of the metric set to apply during computation.
        - relevant_annot_labels (list, optional): List of annotation labels to analyze.
                                                  If `None`, the entire EEG file is used.
                                                  If `['all']`, all annotations will be used.
        - ep_dur (int, optional): Duration of each epoch in seconds.
        - ep_start (int, optional): Start offset for epoching in seconds,
                                    relative to the EEG/annotation.
        - ep_stop (int, optional): Maximum duration of the analyzed segment in seconds.
        - overlap (int, optional): Amount of overlap between epochs in seconds. Defaults to 0.
        - task_label (str, optional): Task label to use for epoching if the whole file is analyzed.

        Returns:
        - pandas.DataFrame: A DataFrame containing metrics for each channel across all
                            epochs in the EEG/annotated segments.

        Notes:
        - If `relevant_annot_labels` is provided with `['all']`, metrics are calculated
          for all annotations in the file.
        - If `relevant_annot_labels` contains specific annotations, only those are used.
          Otherwise, metric computation defaults to the entire EEG file.
        """
        # Check if annotation labels are provided
        if relevant_annot_labels:
            if relevant_annot_labels[0] == 'all':
                # Use all annotations if label 'all' is provided
                full_results_frame = self.calc_metric_from_annotations(
                    metric_name, ep_dur, ep_start, ep_stop, overlap, None
                )
            else:
                # Use only the annotations specified in relevant_annot_labels
                full_results_frame = self.calc_metric_from_annotations(
                    metric_name, ep_dur, ep_start, ep_stop, overlap, relevant_annot_labels
                )
        else:
            # If no annotation labels are provided, process the entire file
            full_results_frame = self.calc_metric_from_whole_file(
                metric_name, ep_dur, ep_start, ep_stop, overlap, task_label
            )

        return full_results_frame


    ########################################################################################################################
    ######################################## high level functions ##########################################################
    ########################################################################################################################

    def compute_metrics(self, metric_set_name: str, annot: list, outfile: str, lfreq: int, hfreq: int,
                        montage: str, ep_start: int = None, ep_stop: int = None, ep_dur: int = None, overlap : int = 0,
                        resamp_freq=None, repeat_measurement: bool = False, include_chaos_pipe=True, multiprocess: bool = False) -> str:
        """
        Compute metrics for EEG processing

        using filtering, re-montaging, resampling, and metric calculations on individual EEG channels
        This is the primary function for processing and analyzing EEG data.

        Args:
        - metric_set_name (str): Name of the metric set to calculate.
        - annot (list): List of annotations to use. Should match the names in the infile annotations.
                        If not provided, will use all annotations with a positive duration.
        - outfile (str): File path where the resulting metrics (CSV) will be saved.
        - lfreq (int): High-pass frequency cutoff for filtering data before metric calculations.
        - hfreq (int): Low-pass frequency cutoff for filtering data before metric calculations.
                       The filter allows frequencies between lfreq and hfreq to pass.
        - montage (str): Name of the montage to apply. Valid options are:
                         'avg', specific reference channel, 'doublebanana', 'circumferential'.
        - ep_start (int, optional): Start offset for epoching in seconds, relative to the beginning
                                     of the file or annotation. Defaults to 0.
        - ep_stop (int, optional): Stop offset for epoching in seconds, relative to the beginning
                                    of the file or annotation. Defaults to the length of the file or annotation.
        - ep_dur (int, optional): Duration of individual epochs in seconds. Defaults to the length
                                   of the file or annotation.
        - overlap (int, optional): Amount of overlap between epochs in seconds. Defaults to 0.
        - resamp_freq (int, optional): Frequency to which the data will be downsampled. Defaults to None
                                       (no downsampling).
        - repeat_measurement (bool, optional): If True and the metrics CSV file already exists, the
                                               calculation is redone, and the existing file is overwritten.
                                               If False, existing metrics are reused, and computation is skipped.
        - include_chaos_pipe (bool, optional): If True, includes the pipeline by Toker. Requires a valid
                                               MATLAB version with the pipeline accessible in its path.
        - multiprocess (bool, optional): If True, enables multiprocessing for metric computations. Defaults to False.

        Returns:
        - str: A message indicating the outcome of the processing. Possible messages:
               * 'finished and saved successfully': When computation and saving succeed.
               * 'no metrics could be calculated': When no metrics are computed.
               * Other messages depending on the checks performed in the function.

        Outputs:
        - Saves a CSV file with computed metrics at the specified `outfile` location.
        """

        try:
            # check the name of the outfile
            outfile_check, outfile_check_message = self.buttler.check_outfile_name(outfile,
                                                                                   file_exists_ok=repeat_measurement)
            if not outfile_check:
                return outfile_check_message

            # only keeps channels which correspond to the typical 10-20 system names
            bipolar = self.only_keep_10_20_channels_and_check_bipolar()
            if bipolar:
                print(f'Most likely already has a bipolar montage \n'
                      f'Channel names: \n {self.raw.ch_names}')

            # filter
            self.apply_filter(lfreq, hfreq)

            # downsample
            self.downsample(resamp_freq)

            # montage (also excludes bads and non-EEG channels even if no remontaging is done)
            raw = self.change_montage(montage)
            if not raw:
                return 'could not set montage, maybe EEG is faulty, skipping EEG'
            else:
                self.raw = raw

            # plots for debugging
            # raw.plot(block=True)

            # extract the task label in case only epoching is used to use as annot
            task_label = self.buttler.find_task_from_filename(self.datapath)

            # calculate the metrics
            full_results_frame = self.compute_metrics_fif(
                metric_set_name, annot, ep_dur, ep_start, ep_stop, overlap
            )

            # save dataframe to csv
            if not full_results_frame.empty:
                full_results_frame.to_csv(outfile)
                return 'finished and saved successfully'
            else:
                return 'no metrics could be calculated'
        except Exception as e:
            return f'Error during metric computation: {str(e)}'
