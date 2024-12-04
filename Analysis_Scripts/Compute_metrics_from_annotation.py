import argparse
import mne
import pandas as pd
import os
import numpy as np
from icecream import ic
import Metrics

# This file is meant to load a raw.fif file and compute the desired metrics and save them to csv afterward
# it is used within the apply_script_to_... files

########################################################################################################################
########################################## load and process data #######################################################
########################################################################################################################
def load_data_file(data_file: str):
    '''
    Uses mne to load a readable file format into a raw instance and extracts its sampling frequency
    input:
    - data_file: datapath of the file you want to load (required)
    outputs:
    - raw: raw instance from mne
    - sfreq: sampling frequency of the EEG
    '''
    try:
        raw = mne.io.read_raw(data_file, preload=True)
        sfreq = raw.info['sfreq']
        return raw, sfreq
    except Exception as e:
        print(f'!!!!!!!!!!!!!!!!! could not load file see error below !!!!!!!!!!!!!!!!!!!!!  \n {e}')
        return None, None


def apply_filter(raw: mne.io.Raw, l_freq: float = None, h_freq: float = None, picks: str = 'eeg'):
    '''
    filters a raw instance and returns it afterwards, can be used for lowpass, highpass, bandpass, bandstop filter
    inputs:
    -raw: raw instance (required)
    -l_freq: the lower end of frequencies that should be passed, so essentially only frequencies above it will be included
    -h_freq: the higher end of frequencies that should be passed, so essentially only frequencies below it will be included
    -picks: makes sure only eeg channels are used which can be helpful from time to time
    outputs:
    -None, instead alters the inputed raw instance
    '''
    if l_freq and l_freq != 'None' and h_freq and h_freq != 'None':
        raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks)
    elif l_freq and l_freq != 'None':
        raw.filter(l_freq=l_freq, h_freq=raw.info['sfreq'], picks=picks)
    elif h_freq and h_freq != 'None':
        raw.filter(l_freq=0, h_freq=l_freq, picks=picks)


def only_keep_10_20_channels_and_check_bipolar(raw):
    '''
    Checks the EEG channels for containing a valid part, only once and no part that is marked as invalid in order to
    only keep the correct EEG channels and marke others as bad
    returns True if two valid channel parts are in one channel
    inputs:
    -raw: raw instance from EEG
    outputs:
    -duplicate_positive: returns True if a bipolar montage is used as the channel names contain two channels
    '''
    valid_channel_parts = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                           'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    invalid_channel_parts = ['pO2', 'CO2', 'X', 'Res', 'time']
    added_bads = []
    duplicate_positive = False
    for ch in raw.ch_names:
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
    raw.info['bads'] = raw.info['bads'] + added_bads
    raw.info['bads'] = list(set(raw.info['bads']))
    return duplicate_positive


def convert_electrode_names_to_channel_names(electrode_names: list[str], channel_names: list[str]):
    '''
    Goes through the electrode names and converts them to channel names if the electrode name is part of the channel name
    inputs:
    -electrode_names: names of the electrodes from the montage
    -channel_names: names of the channels after conversion to ensure unified names in the outputs
    returns:
    -outputs_array: array of the same size as electrode_names containing either None or the according channel name
    '''
    output_array = list()
    for i, e_name in enumerate(electrode_names):
        output_array.append(None)
        for c_name in channel_names:
            if e_name.lower() in c_name.lower():
                output_array[i] = c_name
                break
    return output_array


def ensure_electrodes_present(anodes, cathods, new_names):
    '''
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
    '''
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



def change_montage(raw: mne.io.Raw, montage: str):
    '''
    changes the montage of a raw instance
    input:
    -raw: raw instance of the EEG
    -montage: name of the montage, either 'avg', 'doublebanana', 'circumferential' or a specific channel
    returns:
    -raw_internal: the changed raw instance, can be None if there is an error during computation
    '''
    # in this step now automatically some channels are excluded if marked as bad or not marked as EEG channels
    raw_internal = raw.pick(exclude='bads', picks='eeg').copy()
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
        anode_eeg_channels = convert_electrode_names_to_channel_names(anodes, raw_internal.ch_names)
        cathode_eeg_channels = convert_electrode_names_to_channel_names(cathodes, raw_internal.ch_names)
        try:
            # try setting the new bipolar reference
            ic(mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                         ch_name=new_names, copy=False))
        except ValueError:
            try:
                # if setting failes in the previous step we need to check all channels are present and match
                anode_eeg_channels, cathode_eeg_channels, new_names, dropped_names = ensure_electrodes_present(
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
        anode_eeg_channels = convert_electrode_names_to_channel_names(anodes, raw_internal.ch_names)
        cathode_eeg_channels = convert_electrode_names_to_channel_names(cathodes, raw_internal.ch_names)
        try:
            # set montage
            mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                      ch_name=new_names, copy=False)
        except ValueError:
            # drop channels if montage could not be dropped
            anode_eeg_channels, cathode_eeg_channels, new_names, dropped_names = ensure_electrodes_present(
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


def check_outfile_name(outfile: str, file_exists_ok: bool = True):
    '''
    utility function which checks if the output file allready exists and if the path is valid
    inputs:
    -outfile: path where results should be saved
    -file_exists_ok: if the file can allready exists or if that should give an error
    returns:
    -1 if everything with the file path is ok and 0 if there is a problem with the path
    '''
    # check that outputfile is correct and ensure that the path exists
    if outfile.endswith('metrics.csv'):
        dirpath = os.path.dirname(outfile)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
            return 1, 'path was created, everything ok'
    else:
        return 0, 'name not valid, check if it ends with _metrics.csv'

    # check if file already exists, if so return 'file already exists'
    if os.path.exists(outfile) and not file_exists_ok:
        return 0, 'file already exists'
    return 1, 'everything ok'


def find_task_from_filename(filename):
    '''
    uses the filename to extract a task from it, essentially needs the keyword task in the name of the file and extracts
    '***_task-[extracts this]_***'
    input:
    -filename: name of the file to extract task from
    return:
    -task: name of the task that was extracted
    '''
    task: str = None
    file_name_list = filename.split('_')
    for name_part in file_name_list:
        if 'task' in name_part:
            task = name_part.split('-')[1:]
    if isinstance(task, list):
        task = '-'.join(task)
    return task


def map_chaos_pipe_result_to_float(result: str) -> float:
    '''
    only relevant for the 0-1 chaos pipeline results from tokers original matlab implementation to convert a string to
    float
    '''
    # map the output of the chaos pipeline result to a float in order to have only floats in the dataframe
    if result == 'periodic':
        return 0
    if result == "chaotic":
        return 1
    if result == 'stochastic':
        return 2
    if result == 'nonstationary':
        return 3
    else:
        return 4  # something went wrong here


########################################################################################################################
############################################# metric calculation #######################################################
########################################################################################################################

############################################ basic functions ###########################################################


def initialize_metric_functions(name):
    '''
    uses the Metrics initialized from Metrics.py and loads the functions, names, and kwargs for them
    inputs:
    -name: name of the metrics set that should be loaded
    returns:
    -metrics_functions: the list of functions to be calculated on the timeseries
    -metrics_name_list: list of names for the functions used to save the results
    -kwargs_list: list of dictionaries with additional arguments for the functions
    '''
    metrics_functions, metrics_name_list, kwargs_list = Metrics.select_metrics(name)
    return metrics_functions, metrics_name_list, kwargs_list


def apply_metric_func(eeg, metric_func, kwargs):
    '''
    applies a function to a timeseries (eeg channel)
    inputs:
    -eeg: channel data of an eeg channel (one dimensional time series)
    -metric_func: function which is calculated based on the eeg
    -kwargs: additional arguments for the function
    returns:
    -function ouput after calculation on the eeg
    '''
    # print(f'{datetime.today().strftime("%Y-%m-%d %H:%M:%S")} : Applying function {metric_func.__name__}'
    #       f' with hyperparameters: {kwargs}')

    # ensures eeg channel is saved as contiguos array in memory
    eeg = np.ascontiguousarray(eeg)
    if kwargs:
        try:
            # try applying kwargs as kwargs
            return metric_func(eeg, **kwargs)
        except TypeError:
            # if kwargs are not accepted use kwarg values as args list and try again
            return metric_func(eeg, *list(kwargs.values()))
        except Exception as e:
            print(f'could not apply metric {metric_func.__name__} to eeg with exception {e}')
    else:
        # if no kwargs are given just calculate with default parameters
        return metric_func(eeg)


def extract_eeg_columns(eeg_dataframe):
    '''
    Extracts all besides the first column from the eeg dataframe
    inputs:
    -dataframe with eeg data
    returns:
    -list of channels
    '''
    return eeg_dataframe.columns[1:]


def create_result_array(eeg_np_array, metrics_func_list: list, kwargs_list: list[dict]) -> list:
    '''
    Creates a list of results using the function list and eeg data
    inputs:
    -eeg_np_array: array converted to numpy standard from the eeg frame
    -metrics_func_list: list of functions to compute on eeg
    -kwargs_list: list of additional arguments for the metrics
    returns:
    -result_list: list that contains the results per metric for the eeg data
    '''
    result_list = []
    for metric_func, kwargs in zip(metrics_func_list, kwargs_list):
        result_list.append(apply_metric_func(eeg_np_array, metric_func, kwargs))
    return result_list


############################################ advanced functions ########################################################


def process_result_array(result_array: list, metric_name_array: list[str]) -> list:
    '''
    The results from the metrics are often returned as lists or dictionaries with additional parameters which we dont
    want to have in the csv
    inputs:
    results_array: array of results that can be of type dict or list
    metric_name_array: array of names for the metrics in the csv
    returns:
    processed_array: array with (name, metric) tuples and extracted from the original form
    '''
    # only keep the relevant metric parts for further processing
    processed_array = []
    for result in result_array:
        result_type = type(result)
        if result_type == list or result_type == tuple:
            processed_array.append(result[0])
        elif result_type == dict:
            for key, value in result.items():
                if key == 'result':
                    value = map_chaos_pipe_result_to_float(value)
                processed_array.append(value)
        else:
            processed_array.append(result)
    # create tuples with names
    for i, (name, pai) in enumerate(zip(metric_name_array, processed_array)):
        processed_array[i] = (name, pai)
    return processed_array


def create_result_dict_from_eeg_frame(eeg_dataframe, metrics_func_list: list,
                                      metrics_name_list: list[str], kwargs_list: list[dict], channelwise=True) -> (dict, list[str]):
    '''
    Creates a metric frame from the combination of the eeg and the metric lists
    input:
    eeg_dataframe: dataframe with the eeg data
    metrics_func_list: list of the metric functions
    metrics_name_list: list of names of the metric functions
    kwargs_list: list of dictionaries with additional arguments
    channelwise: if metrics should be computet per single timeseries or on the full frame
    returns:
    result_dict: dictionary with keys beeing the eeg columns and values the computed metrics
    metrics_name_list: list of the metric names
    '''
    result_dict = {}
    eeg_cols = extract_eeg_columns(eeg_dataframe)
    if channelwise:
        for column in eeg_cols:
            eeg = eeg_dataframe[column].to_numpy()
            if column == 'time':
                # time_vec, should not be strictly necessary to do this check anymore theoretically
                continue
            raw_result_array = create_result_array(eeg, metrics_func_list, kwargs_list)
            processed_result_array = process_result_array(raw_result_array, metrics_name_list)
            result_dict[column] = processed_result_array
    else:
        eeg = eeg_dataframe.to_numpy()
        raw_result_array = create_result_array(eeg, metrics_func_list, kwargs_list)
        processed_result_array = process_result_array(raw_result_array, metrics_name_list)
        result_dict = {column: processed_result_array for column in eeg_cols}
        #pprint(result_dict, indent=4)
    return result_dict, metrics_name_list


def create_dataframe_from_result_dict(result_dict: dict, metric_name_array: list[str],
                                      start_data_record: float, duration: float, label: str) -> pd.DataFrame:
    '''
    Takes in result dictionary with some additional parameters to create the dataframe
    input:
    result_dict: dictionary from create_result_dict_from_eeg_frame with eeg columns and the according values
    metric_name_array: array of names for the lists provided in the above dictionary
    start_data_record: float containing the start time of the eeg
    duration: duration of the eeg segment
    label: label of the eeg
    returns:
    sub_results_frame:
    '''
    # create a multiindexing based on metric, label, startDataRecord, duration
    index = pd.MultiIndex.from_product([[label], [start_data_record], [duration], metric_name_array],
                                       names=['label', 'startDataRecord', 'duration', 'metric'])
    eeg_column_names = list(result_dict.keys())
    sub_results_frame = pd.DataFrame(columns=eeg_column_names,
                                     index=index,
                                     dtype=float)
    # put the results into the dataframe
    for column, result_array in result_dict.items():
        for result_tuple in result_array:
            result = result_tuple[1]
            metric_name = result_tuple[0]
            sub_results_frame.loc[(label, start_data_record, duration, metric_name), column] = result
    return sub_results_frame


def calc_metrics_from_eeg_dataframe_and_annotations(eeg_dataframe: pd.DataFrame, metric_set_name,
                                                    annot_label: str, annot_startDataRecord: float,
                                                    annot_duration: float) -> pd.DataFrame:
    '''
    Function which combines sub steps of computing metrics, processing the results and creating a sub dataframe from
    the metrics which can be later combined to a larger frame.
    input:
    - eeg_dataframe: pandas frame with eeg data to be analyzed
    - metric_set_name: name of the metric set
    - annot_label: label of the annotation of the eeg
    - annot_startDataRecord: time of the start of the annotation
    - annot_duration: duration of the annotated segment to be analyzed
    returns:
    - sub_results_frame: dataframe containing the metrics per channel for the segment
    '''
    # initialize the metrics which should be calculated
    metrics_functions, metrics_name_list, kwargs_list = initialize_metric_functions(metric_set_name)
    # calculate the results for the metrics and store them to dict
    result_dict, metrics_name_list = create_result_dict_from_eeg_frame(eeg_dataframe, metrics_functions,
                                                                       metrics_name_list, kwargs_list)
    # create the sub_results_frame from results dict
    sub_results_frame = create_dataframe_from_result_dict(result_dict, metrics_name_list, annot_startDataRecord,
                                                          annot_duration, annot_label)
    return sub_results_frame


def epoching(raw: mne.io.Raw, metric_set_name, duration: int = None, start_time: int = None, stop_time: int = None,
             overlap: int = 0, task: str = None) -> pd.DataFrame:
    '''
    instead of using annotations in the data constant epochs are used
    duration, start_time and stop_time need to be given in seconds. Only duration is mandatory.
    epochs are created in the interval [start, stop) with steps of duration
    only full duration intervals are created
    '''
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    full_epoch_frame = pd.DataFrame()
    # set default parameters
    if not stop_time:
        stop_time = np.round(n_samples / sfreq)
    else:
        stop_time = int(min(np.round(n_samples / sfreq), stop_time))
    if not start_time:
        start_time = 0
    if not duration:
        duration = int(np.floor(stop_time - start_time))
    if overlap:
        if overlap >= duration:
            overlap = 0
            print('overlap cant be bigger or equal to the duration, was reset to 0')
    # go through the different epochs
    for t_onset in np.arange(start_time, (stop_time - duration) + 1, duration-overlap):
        t_onset_samples = t_onset * sfreq
        t_stop_samples = (t_onset + duration) * sfreq
        eeg_dataframe = raw.to_data_frame(
            start=int(t_onset_samples),
            stop=int(t_stop_samples))
        # calculat the results of the current epoch and save them to dataframe
        print(f'Calculating for times: {t_onset}:{t_onset + duration}')
        sub_results_frame = calc_metrics_from_eeg_dataframe_and_annotations(eeg_dataframe, metric_set_name,
                                                                            task, t_onset, duration)
        # add the calculated metrics to the frame with all the other epochs
        full_epoch_frame = pd.concat([full_epoch_frame, sub_results_frame], axis=0)
    return full_epoch_frame


def calc_metric_from_annotations(raw: mne.io.Raw, metric_set_name, ep_dur: int, ep_start: int, ep_stop: int,
                                 overlap: int = 0, relevant_annot_labels: list = None) -> pd.DataFrame:
    '''
    calculates a dataframe for the full annotation epoched into segments.
    inputs:
    - raw: the raw object of the eeg
    - metrics_set_name: name of the metric set
    - ep_dur: duration of the epoch
    - ep_start: start of the epoch relative to eeg/annotation
    - ep_stop: maximum duration of the analyzed segment
    - relevant_annot_labels: list of annotations that should be analyzed, if None whole eeg is used
    returns:
    - full_annot_frame: pandas dataframe with metrics per channel for all epochs in an eeg/annotated eeg segment
    '''
    if ep_start is None:
        ep_start = 0
    raw_annots = raw.annotations
    full_annot_frame = pd.DataFrame()
    # chec that the file actually has annotations
    if raw_annots:
        for annot in raw_annots:
            annot_name = annot['description']
            # if relevant annot labels are given select for them
            if relevant_annot_labels:
                if annot_name in relevant_annot_labels:
                    # extract start and duration of annotation
                    annot_start_seconds = annot['onset']
                    annot_duration_seconds = annot['duration']
                    annot_stop_seconds = annot_start_seconds + annot_duration_seconds
                    print(f'annotation: {annot_name}, times: {annot_start_seconds}:{annot_stop_seconds}')
                    # use the epoching info to calculate the final times
                    ep_start_seconds = annot_start_seconds + ep_start
                    if ep_stop:
                        ep_stop_seconds = min(ep_start_seconds + ep_stop, annot_stop_seconds)
                    else:
                        ep_stop_seconds = annot_stop_seconds
                    # call the epoching function to calc the metrics
                    sub_results_frame = epoching(raw, metric_set_name, ep_dur, ep_start_seconds, ep_stop_seconds,
                                                 overlap, annot_name)
                    # add the calculated metrics to the frame with all the annotations
                    full_annot_frame = pd.concat([full_annot_frame, sub_results_frame], axis=0)

            # if no relevant annot labels are given use all annots
            else:
                # extract start and duration of annotation
                annot_start_seconds = annot['onset']
                annot_duration_seconds = annot['duration']
                annot_stop_seconds = annot_start_seconds + annot_duration_seconds
                # use the epoching info to calculate the final times
                ep_start_seconds = annot_start_seconds + ep_start
                ep_stop_seconds = min(ep_start_seconds + ep_stop, annot_stop_seconds)
                # call the epoching function to calc the metrics
                sub_results_frame = epoching(raw, metric_set_name, ep_dur, ep_start_seconds, ep_stop_seconds,
                                             overlap, annot_name)
                # add the calculated metrics to the frame with all the annotations
                full_annot_frame = pd.concat([full_annot_frame, sub_results_frame], axis=0)
    return full_annot_frame


def compute_metrics_fif(raw: mne.io.Raw, metric_set_name, relevant_annot_labels: list = None,
                        ep_dur=None, ep_start=None, ep_stop=None, overlap : int  = 0,
                        task_label=None) -> pd.DataFrame:
    '''
    wrapper functions around calc_metric_from_annotation or epoching to deal with eegs with one, multiple or no annotation
    inputs:
    - raw: the raw object of the eeg
    - metrics_set_name: name of the metric set
    - ep_dur: duration of the epoch
    - ep_start: start of the epoch relative to eeg/annotation
    - ep_stop: maximum duration of the analyzed segment
    - relevant_annot_labels: list of annotations that should be analyzed, if None whole eeg is used
    returns:
    - full_result_frame: pandas dataframe with metrics per channel for all epochs in an eeg/annotated eeg segment
    '''
    # annotation labels provided by user
    if relevant_annot_labels:
        # all annotations should be used
        if relevant_annot_labels[0] == 'all':
            full_results_frame = calc_metric_from_annotations(raw, metric_set_name, ep_dur, ep_start, ep_stop, None)
        # only the annotations in the relevant_annot_labels should be used
        else:
            full_results_frame = calc_metric_from_annotations(raw, metric_set_name, ep_dur, ep_start, ep_stop,
                                                              relevant_annot_labels)
    # no annotation labels provided by user, whole file will be used
    else:
        full_results_frame = epoching(raw, metric_set_name, ep_dur, ep_start, ep_stop, overlap, task_label)
    return full_results_frame


########################################################################################################################
######################################## high level functions ##########################################################
########################################################################################################################
def compute_metrics(infile_data: str, metric_set_name: str, annot: list, outfile: str, lfreq: int, hfreq: int,
                    montage: str, ep_start: int = None, ep_stop: int = None, ep_dur: int = None, overlap : int = 0,
                    resamp_freq=None, repeat_measurement: bool = False, include_chaos_pipe=True, multiprocess: bool = False) -> str:
    """
    Compute the metrics for the input file per timeseries and save the results to csv.
    Metrics which are computed:
    - fractal dimension (katz algorithm)
    - permutation entropy
    - lempel ziv complexity
    - largest lyapunov exponent
    - chaos pipeline by toker (using matlab engine)
    inputs:
    - infile_data: the file containing the eeg for which the metrics are to be calculated
    - annot: the annotations which should be used. Needs to contain names equal to the ones in the infile_annot.
                If not provided will use all annotations which have a positive duration.
    - outfile: the file to which the csv with the metrics is saved
    - lfreq: highpass freq for filtering the data before metrics calc
    - hfreq: lowpass freq for filtering the data before metrics calc
        A filter is designed i a way that it allows frequencies inbetween low and high to pass
    - montage: string which defines the montage for the calculation valid options are: 'avg', refchannel, 'doublebanana',
                'circumferential'
    - ep_start: start offset for the epoching, defaults to 0, needs to be given in seconds after beginning of file/annot
    - ep_stop: stop offset for the epoching, defaults to length of file/annot, needs to be given in seconds
    - ep_dur: duration of the epochs, defaults to length of file/annot, needs to be given in seconds
    - repeat_measurement: boolean which is given to out_file_check. If True and the metrics.csv file allready exists it
                will be recalculated and overwritten, if False the metrics calculation will be skipped and the original
                file will persist
    - include_chaos_pipe: boolean if the pipleine by toker should be used. Requires a matlab version with the pipeline
                        on its path
    outputs:
    - saves a csv with the metrics to the outfile location
    - returns a string informing about what was done by the function
    """
    # check the name of the outfile
    outfile_check, outfile_check_message = check_outfile_name(outfile, file_exists_ok=repeat_measurement)
    if not outfile_check:
        return outfile_check_message

    # load data
    raw, sfreq = load_data_file(infile_data)

    if not raw:
        return 'EEG data could not be loaded, skipping EEG'


    # only keeps channels which correspond to the typical 10-20 system names
    bipolar = only_keep_10_20_channels_and_check_bipolar(raw)
    if bipolar:
        print(f'Most likely allready has a bipolar montage \n'
              f'Channel names: \n {raw.ch_names}')

    # filter
    apply_filter(raw, lfreq, hfreq)

    # downsample
    if raw.info['sfreq'] > resamp_freq and resamp_freq:
        raw.resample(resamp_freq)

    # montage (also excludes bads and non eeg channels even if no remontaging is done)
    raw = change_montage(raw, montage)

    if not raw:
        return 'could not set montage, maybe EEG is faulty, skipping EEG'

    # plots for debugging
    # raw.plot(block=True)

    # extract the task label incase only epoching is used to use as annot
    task_label = find_task_from_filename(infile_data)

    # calculate the metrics
    full_results_frame = compute_metrics_fif(raw, metric_set_name, annot, ep_dur, ep_start, ep_stop, overlap,
                                             task_label=task_label)

    # save dataframe to csv
    if not full_results_frame.empty:
        full_results_frame.to_csv(outfile)
        return 'finished and saved successfully'
    else:
        return 'no metrics could be calculated'


def check_file_exists_and_create_path(log_file):
    '''
    checks if a file path exists (not the file itself) and makes sure the path is created
    inputs:
    - log_file: the logfile name with path
    returns:
    - Nothing
    '''
    if os.path.dirname(log_file) and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)


def main():
    '''
    main function of the file, parsing arguments, creating logfile and computing metrics for one eeg. Only used if
    the file itself is called via commandline.
    '''
    parser = argparse.ArgumentParser(prog='Compute_metrics_from_raw_file',
                                     description='Takes a *raw.fif file and computes metrics from it'
                                                 ' which are saved to csv')
    parser.add_argument('--input_file', type=str, help='name of input raw.fif file')
    parser.add_argument('--annot_names', type=str, nargs='?', const=None,
                        help='names of annotations that should be used')
    parser.add_argument('--output_file', type=str, help='file where to save the computed metrics,'
                                                        ' needs to end with metrics.csv')
    parser.add_argument('--lfreq', type=int, help='lowe edge frequency for filtering')
    parser.add_argument('--hfreq', type=int, help='high edge frequency for filtering')
    parser.add_argument('--montage', type=str, help='montage to be used. Valid options are'
                                                    ' either a channel from the data or one of the following:'
                                                    ' "avg", "doublebanana", "circumferential')
    parser.add_argument('--ep_start', type=int, help='start offset for epoching [s]')
    parser.add_argument('--ep_dur', type=int, help='duration of one epoch [s]')
    parser.add_argument('--ep_stop', type=int, help='stop offset for epoching [s]')
    parser.add_argument('--ep_overlap', type=int, help='overlap to use for sliding window [s]')
    parser.add_argument('--sfreq', type=int, help='will resample to this freq'
                                                  ' if the current sample frequency is higher.'
                                                  ' No upsampling will be done')
    parser.add_argument('--metrics', type=str, help='Needs to be set to one of the values available'
                                                    'in Metrics.py')
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    annot_names = args.annot_names
    lfreq = args.lfreq
    hfreq = args.hfreq
    montage = args.montage
    ep_start = args.ep_start
    ep_dur = args.ep_dur
    ep_stop = args.ep_stop
    overlap = args.ep_overlap
    resamp_freq = args.sfreq
    metric_set_name = args.metrics
    return compute_metrics(input_file, metric_set_name, annot_names, output_file, lfreq, hfreq, montage,
                           ep_start, ep_stop, ep_dur, overlap, resamp_freq,
                           repeat_measurement=True)


if __name__ == '__main__':
    ic(main())
