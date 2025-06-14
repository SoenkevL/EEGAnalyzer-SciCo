import os.path
import re
from difflib import SequenceMatcher
from pprint import pprint

import mne
import neurokit2 as nk
import mat4py
import numpy as np
from icecream import ic
import pandas as pd

def print_all_builtin_montages():
    builtin_montages = mne.channels.get_builtin_montages(descriptions=True)
    for montage_name, montage_description in builtin_montages:
        print(f"{montage_name}: {montage_description}")

def make_montage(montage_name):
   return mne.channels.make_standard_montage(montage_name)

def show_example_montage(montage_name):
    montage = make_montage(montage_name)
    print(montage)
    montage.plot()  # 2D
    fig = montage.plot(kind="3d", show=False)  # 3D
    fig.gca().view_init(azim=70, elev=15)  # set view angle for tutorial
    fig.show()
    return montage

#TODO: check Ai written code for matching of channel names and see if i should fuse/remove old functions, also check the code
def normalize_electrode_name(name):
    """Extract and normalize electrode name from various formats"""
    # Remove common prefixes (case insensitive)
    name = re.sub(r'^(eeg\s*|e\s*)', '', name.strip(), flags=re.IGNORECASE)
    # Remove any remaining whitespace and convert to standard case
    name = name.strip()
    return name


def find_best_match(raw_name, montage_list, similarity_threshold):
    """Find the best matching electrode from montage list"""
    normalized_raw = normalize_electrode_name(raw_name)

    # First try exact match (case insensitive)
    for electrode in montage_list:
        if normalized_raw.lower() == electrode.lower():
            return electrode

    # Then try fuzzy matching
    best_match = None
    best_score = 0

    for electrode in montage_list:
        # Calculate similarity
        score = SequenceMatcher(None, normalized_raw.lower(), electrode.lower()).ratio()
        if score > best_score and score >= similarity_threshold:
            best_score = score
            best_match = electrode

    return best_match


def create_electrode_mapping(montage_electrodes, raw_channel_names, similarity_threshold=0.8):
    """
    Create a flexible mapping between montage electrodes and raw channel names.

    Parameters:
    -----------
    montage_electrodes : list
        List of standard electrode names from montage
    raw_channel_names : list
        List of channel names from raw data
    similarity_threshold : float
        Minimum similarity score for fuzzy matching (0-1)

    Returns:
    --------
    dict : mapping from raw channel names to montage electrode names
    """
    # Create the mapping
    mapping = {}
    unmatched = []

    for raw_name in raw_channel_names:
        matched_electrode = find_best_match(raw_name, montage_electrodes, similarity_threshold)
        if matched_electrode:
            mapping[raw_name] = matched_electrode
        else:
            unmatched.append(raw_name)

    # Print results
    print("Electrode Mapping:")
    print("-" * 50)
    for raw_name, electrode in mapping.items():
        print(f"'{raw_name}' -> '{electrode}'")

    if unmatched:
        print(f"\nUnmatched channels ({len(unmatched)}):")
        for channel in unmatched:
            print(f"  {channel}")

    print(f"\nTotal matched: {len(mapping)}/{len(raw_channel_names)}")

    return mapping, unmatched

def test_create_electrode_mapping():
    # Your electrode lists
    montage_electrodes = ['Fp1', 'Fpz', 'Fp2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', 'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', 'O1', 'Oz', 'O2', 'O9', 'Iz', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2']

    raw_channel_names = ['EEG Fp1', 'EEG F3', 'EEG C3', 'EEG P3', 'EEG O1', 'EEG F7', 'EEG T3', 'EEG T5', 'EEG Fc1', 'EEG Fc5', 'EEG Cp1', 'EEG Cp5', 'EEG F9', 'EEG Fz', 'EEG Cz', 'EEG Pz', 'EEG Fp2', 'EEG F4', 'EEG C4', 'EEG P4', 'EEG O2', 'EEG F8', 'EEG T4', 'EEG T6', 'EEG Fc2', 'EEG Fc6', 'EEG Cp2', 'EEG Cp6', 'EEG F10']

    # Create the mapping
    electrode_mapping = create_electrode_mapping(montage_electrodes, raw_channel_names)

    # Test with additional examples you mentioned
    test_names = ['eeg F8', 'e F8', 'f8', 'eeg f8']
    print(f"\nTesting additional examples:")
    print("-" * 30)
    for test_name in test_names:
        normalized = re.sub(r'^(eeg\s*|e\s*)', '', test_name.strip(), flags=re.IGNORECASE).strip()
        # Find match in montage
        match = None
        for electrode in montage_electrodes:
            if normalized.lower() == electrode.lower():
                match = electrode
                break
        print(f"'{test_name}' -> '{match}' (normalized: '{normalized}')")

    # Get the rename dictionary
    rename_dict = apply_electrode_mapping(None, electrode_mapping)
    print(f"\nRename dictionary for MNE:")
    pprint(rename_dict)

# Function to apply mapping to rename channels
def apply_electrode_mapping(raw_object, mapping):
    """
    Apply the electrode mapping to rename channels in a raw object.
    This assumes you're using MNE-Python or similar library.
    """
    # Create rename dictionary (only for matched channels)
    rename_dict = {old_name: new_name for old_name, new_name in mapping.items()}

    # If using MNE-Python, you would do:
    # raw_object.rename_channels(rename_dict)

    return rename_dict


### end new code
def only_keep_10_20_channels_and_check_bipolar(raw):
    """
    Checks the EEG channels for containing a valid part, only once and no part that is marked as invalid in order to
    only keep the correct EEG channels and marke others as bad
    returns True if two valid channel parts are in one channel name
    inputs:
    - raw: raw instance of the EEG
    returns:
    - duplicate_positives: boolean that is true if a channel has two valid parts indicating a bipolar or circular montage
    """
    valid_channel_parts = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                           'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    invalid_channel_parts = ['pO2', 'CO2', 'X', 'SaO2']
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
    # change the reference used in the eeg file to avg, a bipolar one or a channel/ a list of channels
    if montage == 'avg':
        ic(raw_internal.set_eeg_reference(ref_channels='average'))
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
        anode_eeg_channels = convert_electrode_names_to_channel_names(anodes, raw_internal.ch_names)
        cathode_eeg_channels = convert_electrode_names_to_channel_names(cathodes, raw_internal.ch_names)
        try:
            ic(mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                         ch_name=new_names, copy=False))
        except ValueError:
            try:
                anode_eeg_channels, cathode_eeg_channels, new_names, dropped_names = ensure_electrodes_present(
                    anode_eeg_channels, cathode_eeg_channels, new_names)
                print(
                    f'montage could not be set fully, probably not all needed channels are present. The following channels could not be computed: {dropped_names}')
                ic(mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                             ch_name=new_names, copy=False))
            except ValueError:
                print('montage could not be set at all')
                return None
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
        anode_eeg_channels = convert_electrode_names_to_channel_names(anodes, raw_internal.ch_names)
        cathode_eeg_channels = convert_electrode_names_to_channel_names(cathodes, raw_internal.ch_names)
        try:
            mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                      ch_name=new_names, copy=False)
        except ValueError:
            anode_eeg_channels, cathode_eeg_channels, new_names, dropped_names = ensure_electrodes_present(
                anode_eeg_channels, cathode_eeg_channels, new_names)
            print(
                f'montage could not be set fully, probably not all needed channels are present. The following channels could not be computed: {dropped_names}')
            mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                      ch_name=new_names, copy=False)
    elif montage in raw_internal.ch_names:
        raw_internal.set_eeg_reference(ref_channels=montage)
    else:
        print(f'The given montage is not a viable option or a channel of the raw_internal object, no montage applied')
    return raw_internal


def check_bad_channels_nk(raw):
    '''
    Runs the neurokit implementation of bad channel checking on the raw eeg instance and also sets the index of the
    frame to the names of the channels
    inputs:
    - raw: raw eeg instance
    outputs:
    - bads: bad channel names
    - badchannels_df: dataframe containing more info on why channels are bad
    '''
    # check for bad channels using neurokit
    bads, badchannels_df = nk.eeg_badchannels(raw, show=False, bad_threshold=0.2)
    badchannels_df.index = raw.pick(picks='eeg', exclude='bads').ch_names
    return bads, badchannels_df


def add_channels_to_bads(raw, bad_channel_names, method='add'):
    '''
    Helper function to manipulate bad channels of a raw object. Can either replace or add channels and ensures channels
    are unique
    inputs:
    - raw: raw instance of eeg
    - bad_channel_names: list of names with the bad channels to add/replace
    - method: the method to use, either 'add' or 'replace'
    outputs:
    - raw: the changed raw object
    '''
    if method == 'add':
        raw.info['bads'] = raw.info['bads']+[ch for ch in raw.ch_names if ch in bad_channel_names]
    elif method == 'replace':
        raw.info['bads'] = [ch for ch in raw.ch_names if ch in bad_channel_names]
    raw.info['bads'] = list(set(raw.info['bads']))
    return raw


def update_annotations_suzanne(raw, annot_path, sampleSignalPath, method='add', recompute=False):
    '''
    Python function which uses a matlab script to extract annotations from an edf file. The method needs to have
    matlab set up with the script working and in matlabs default path to work. In case annotations were already
    extracted prior the functions instead uses these besides if recompute is set to true.
    inputs:
    - raw: raw object of the eeg
    - sampleSignalPath: path to the eeg edf file
    - annot_path: path to the annotations (either existing or used as a save path)
    - method: method how annotations should be added to raw object. Can either be 'add' to add to the ones mne read
              or 'replace' to only keep the matlab annotations
    - recompute: If annotations should be computed again using the script even though there are existing annotations
    outputs:
    - raw: changed raw object
    '''
    if recompute or not os.path.exists(annot_path):
        import matlab.engine
        eng = matlab.engine.start_matlab()
        # extract annotations
        eng.extract_annotations_suzanne(sampleSignalPath, annot_path, nargout=0)
    annotations_sz = mat4py.loadmat(annot_path)
    annotations_sz = annotations_sz['annotations']
    # set new annotations
    if annotations_sz:
        new_annots = mne.Annotations(annotations_sz['startDataRecord'], annotations_sz['duration'], annotations_sz['label'])
        original_annots = raw.annotations  # save the original annotations
        if method == 'add':
            annots = original_annots + new_annots
            raw.set_annotations(annots)  # overwrite the original annotations with the ones from the matlab script
            ic('added new annotations to the old annotations')
        elif method == 'replace':
            raw.set_annotations(new_annots)  # overwrite the original annotations with the ones from the matlab script
            ic('new annotations set')
    else:
        ic('kept original annotations as none could be extracted')
    return raw


def create_continues_annot_from_start_stop_keys(input, start_key, stop_key, new_label, start_time = None):
    '''
    Creates continuos annotations from start stop keys
    As input either the raw object or a raw.annotations object can be given, same type will be returned
    inputs:
    - input: either raw or raw.annotations object
    - start_key: relative to start of input start time of new annot (can be list)
    - stop_key: relative to start on input stop time of new annot (can be list, should be same length as start_key)
    - new_label: label added ot the annotated segments (can only be one label as a string)
    - start_time: absolute start time of measurement of the eeg, can be found using raw.info['meas_date']
    outputs:
    - an annotation object with the new annotations added to it or the raw object with updated annotations
        depending on the given input
    '''
    annot_objet = False
    if isinstance(input, mne.io.Raw) or isinstance(input, mne.io.edf.edf.RawEDF):
        annots = input.annotations
        start_time = input.info['meas_date'].replace(tzinfo=None)
    elif isinstance(input, mne.Annotations):
        annot_objet = True
        annots = input
        if not start_time:
            raise Exception('Need to provide the absolute start time of the file when providing annots.'
                            'Can be found using raw.info["meas_date"]')
    else:
        raise ValueError('input is not an mne annotations or raw object')
    annot_df = annots.to_data_frame()
    starting_points = annot_df.loc[annot_df.description==start_key, 'onset']
    stopping_points = annot_df.loc[annot_df.description==stop_key, 'onset']
    if not len(starting_points) == len(stopping_points):
        print('be carefull, start and stop arrays are not of same length')
    td = []
    for start, stop in zip(starting_points, stopping_points):
        td.append(float((stop-start).total_seconds()))
    data = np.zeros((len(starting_points),2), dtype=float)
    data[:,0] = [float((starting_point-start_time).total_seconds()) for starting_point in starting_points]
    data[:,1] = td
    continuos_annots = pd.DataFrame(data=data, columns=['onset', 'duration'], dtype=float)
    continuos_annots['description'] = new_label
    new_annot_df = annot_df.loc[~((annot_df.description==start_key) | (annot_df.description==stop_key)), :]
    new_annot_df['onset'] = new_annot_df['onset'].apply(lambda x: (x-start_time).total_seconds())
    new_annot_df = pd.concat((new_annot_df, continuos_annots), axis=0)
    continuos_annots = mne.Annotations(new_annot_df['onset'], new_annot_df['duration'], new_annot_df['description'])
    if annot_objet:
        return continuos_annots
    else:
        input.set_annotations(continuos_annots)
        return input


def find_eog_events(raw, thresh) -> list:
    """
    returns eog events in samples, basically similar to thresholding
    inputs:
    - raw: raw eeg object
    - thresh: threshhold for the peak detection thats underlying, can be none to use a default value
    outputs:
    - eog_events: list of possible eog events in the data (not sure about the exact return type rn as I didnt use it in the end)
    """
    channels = raw.pick(picks='eeg', exclude='bads').ch_names
    eog_events = mne.preprocessing.find_eog_events(raw, ch_name=channels, thresh=thresh)
    eog_events = eog_events[:,0]
    return eog_events

def find_flat_channels(bads_df):
    '''
    Function that finds flat channels from the bads dataframe of check_bad_channels_nk
    inputs:
    - bads_df: dataframe returned from the function check_bad_channels_nk
    outputs:
    - empty_channels: list of flat channel names
    '''
    median_mean = abs(bads_df['Mean'].aggregate('median'))
    empty_series = bads_df[abs(bads_df['Mean']) < 0.00001*median_mean]
    empty_channels = list(empty_series.index)
    return empty_channels

def find_optimal_epochs(raw: mne.io.Raw,
                        start: int, stop: int, duration: int, step: int, search_interval: int, window_overlap: int,
                        annot_name, artifact_thresh=None, montage=None,
                        plot=True, verbose=False):
    """
    This function will search for the optimal epochs in the data. It will start at second start and go untill second stop.
    The epoch annotation will have the duration of duration and name annot_name. It uses step to go through the data
    and search for the optimal interval within the new step time +- search_interval with a window overlap of
    window_overlap.
    The criteria for a good or bad epoch are the number of detected eog events in the prefrontal channels aswell as
    outputs from nk.eeg_badchannels and find_flat_channels
    inputs:
    - raw: raw eeg data
    - start: start time where to search for epoch
    - stop: stip time untill to search for epoch
    - duration: epoch length
    - step: step to traverse from start to stop-duration
    - search_interval: interval to search for optimal epoch if step size is bigger than duration (which is intended)
    - window_overlap: how much overlap whichin epochs within a search interval
    - annot_name: what to name the best picked epoch as annotation in the eeg
    - artifact_thresh: threshhold for eeg artifacts, can be None
    - montage: montage of the signal that is desired (will be changed to the value)
    - plot: If optimal epoch should be plotted when function finished, will block further script execution though
    - verbose: boolean that controls verbosity of feedback to the user, turn on when using for first time
    outputs:
    - new_annots: annotations of the raw object now containing a label for the optimal epoch
    """
    # verbosity using ic
    if verbose:
        ic.enable()
    else:
        ic.disable()
    # init data
    raw_internal = raw.copy().pick(picks='eeg', exclude='bads')
    raw_internal.load_data()
    raw_len = len(raw_internal)
    sfreq = raw_internal.info['sfreq']
    raw_len_t = int(raw_len/sfreq)
    # remontage and sanity check
    if montage:
        raw_internal = change_montage(raw_internal, montage)
    # for visual data inspection and adding bad channels
    if plot:
        raw_internal.plot(block=True)
    # check eog events
    eog_events = ic(find_eog_events(raw_internal, thresh=artifact_thresh))
    # checks for input parameters
    if not start:
        start = 0
        ic('start was None, set to zero')
    elif start < 0:
        start = 0
        ic('start was smaller than zero, set to zero')
    if not stop:
        ic('stop was None, set to length of file automatically')
        stop = raw_len_t
    elif stop > raw_len_t:
        ic('stop was beyond file length, set to length of file automatically')
        stop = raw_len_t
    if duration < window_overlap:
        ic('duration can not be smaller than window overlap, set window overlap to 0')
        window_overlap = 0
    if step < duration:
        ic('step can not be smaller than duration, set step to 10*duration')
        step = int(10*duration)
    # prepare annots
    annot_starts = []
    annot_durs = []
    annot_names = []
    # go through the file from start to stop in step
    for idx in np.arange(start, stop-step+1, step):
        # define the search interval for the best epoch
        temp_start = idx - search_interval
        temp_stop = idx + search_interval
        temp_step = duration - window_overlap
        error_scores = []
        starting_points = []
        # sanity checks on the search interval
        if temp_stop > raw_len_t:
            temp_stop = raw_len_t
        if temp_start < 0:
            temp_start = 0
        if temp_start < temp_stop:
            starting_points = np.arange(temp_start, temp_stop-duration+1, temp_step)
            # look for best epoch in the search interval
            for temp_idx in starting_points:
                # extract the epoch from the data
                temp_raw = raw_internal.copy().crop(tmin=temp_idx, tmax=temp_idx+duration)
                # calculate bad channels
                bads, bads_df = check_bad_channels_nk(temp_raw)
                ic(temp_idx)
                ic(bads)
                # calculate number of artifacts in the epoch range
                # find eog events
                eog_art = [e for e in eog_events if temp_idx < e/sfreq < temp_idx+duration]
                ic(eog_art)
                number_of_eog_art = len(eog_art)
                number_of_bads = len(bads)
                # calculate number of flat channels as these are not automatically bad in nk
                flat_ch = find_flat_channels(bads_df)
                ic(flat_ch)
                number_of_flats = len(flat_ch)
                # calculate the error score for each segment using bad channels, eog artifacts and flat channels
                error_score = number_of_eog_art+number_of_bads+number_of_flats
                error_scores.append(error_score)
        try:
            # look for the best window regarding error score in chronological order
            best_window_start = starting_points[error_scores.index(min(error_scores))]
            # raw_internal.plot(duration=int(duration/sfreq), start=int(best_window_start/sfreq), block=True)
            # append to the annotations the best epoch that was found
            annot_starts.append(best_window_start)
            annot_durs.append(duration)
            annot_names.append(annot_name)
        except ValueError:
            print(f'Probably hit end of file at second {idx}')
    # get the original annotations in order to add the new ones
    original_annots = raw.annotations
    original_onset_time = raw.annotations.orig_time
    annots = mne.Annotations(onset=annot_starts, duration=annot_durs, description=annot_names,
                             orig_time=original_onset_time)
    new_annots = annots + original_annots
    return new_annots


def create_subject_num_string(subject_num, max_0s_prepend=2):
    '''
    Subject numbers are given always in three digits, makes sures number adhear to that standard
    and can be adapted to other lenths
    inputs:
    - subject_num: number of the subject as simple int
    - max_0s_prepend: how many zeros should maximally be added, 2 if the overall number length should be three for example
    outputs:
    - string of new subject number with zeros prepended as needed
    '''
    nb_zeros = np.floor(np.log10(subject_num))
    return f'{"0"*int((max_0s_prepend-nb_zeros))}{subject_num}'

def create_patient_info_from_raw_info(infoobject):
    '''
    Returns the sex and age at measurement from the patient info
    inputs:
    - infoobject: raw.info object
    outputs:
    - sex: sex of the patient
    - age_at_meas: age at measurement of the patient
    '''
    meas_date = infoobject['meas_date'] #datetime object
    meas_year = meas_date.year
    meas_month = meas_date.month
    sex = infoobject['subject_info']['sex'] # int
    birthday = infoobject['subject_info']['birthday'] #tuple
    birthday_year, birthday_month, birthday_day = birthday
    age_at_meas = meas_year - birthday_year
    if meas_month > birthday_month:
        age_at_meas = age_at_meas + 1
    return sex, age_at_meas

