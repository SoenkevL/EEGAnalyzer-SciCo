"""
Channel Regex Patterns for EEG Preprocessing

This module contains the default regex patterns used for automatic channel categorization
in EEG preprocessing. These patterns help identify different types of channels based on
their naming conventions.

Author: Soenke van Loh
Date: 2025-06-19
"""

# Define regex patterns for different channel types
DEFAULT_PATTERNS = {
    'EOG': [
        r'(?!.*EEG)(EOG|EYE)',  # Eye movement channels (exclude if contains EEG)
        r'(?!.*EEG)[VH]EOG',  # Vertical/Horizontal EOG (exclude if contains EEG)
        r'(?!.*EEG)Eye',  # Any eye-related channel (exclude if contains EEG)
    ],
    'ECG': [
        r'(?!.*EEG)(ECG|EKG|CARDIAC|HEART)',  # Heart channels (exclude if contains EEG)
    ],
    'EMG': [
        r'(?!.*EEG)(EMG|MUSCLE)',  # Muscle activity (exclude if contains EEG)
    ],
    'SPO2': [
        r'(?!.*EEG)(SPO2|SAT|PULSE)',  # Oxygen saturation (exclude if contains EEG)
    ],
    'RESP': [
        r'(?!.*EEG)(RESP|BREATH|THORAX|CHEST)',  # Respiration (exclude if contains EEG)
    ],
    'STIM': [
        r'(?!.*EEG)(TRIG|STI|EVENT|MARKER|Status)',  # Trigger channels (exclude if contains EEG)
    ],
    'MEG': [
        r'(?!.*EEG)MEG\d+',  # MEG channels (exclude if contains EEG)
        r'(?!.*EEG)MAG\d+',  # Magnetometers (exclude if contains EEG)
        r'(?!.*EEG)GRAD\d+',  # Gradiometers (exclude if contains EEG)
    ],
    'MISC': [
        r'(?!.*EEG)(MISC|OTHER|AUX)',  # Miscellaneous (exclude if contains EEG)
    ],
    'EEG': [
        r'(EEG|E)\d+',  # EEG channels with numbers
        r'[A-Za-z]+\d+',  # Any letters followed by numbers (covers most electrode names)
        r'[A-Za-z]*z\d*',  # Midline electrodes ending in 'z' (with optional numbers)
    ],
}

MNE_CHANNEL_TYPES = ['bio', 'chpi', 'dbs', 'dipole', 'ecg', 'ecog', 'eeg', 'emg',
'eog', 'exci', 'eyetrack', 'fnirs', 'gof', 'gsr', 'ias', 'misc',
'meg', 'ref_meg', 'resp', 'seeg', 'stim', 'syst', 'temperature']

# Function to get the default patterns
def get_default_patterns():
    """
    Returns the default regex patterns for channel categorization.
    
    Returns
    -------
    dict
        Dictionary containing regex patterns for different channel types
    """
    return DEFAULT_PATTERNS.copy()

def get_mne_channel_types():
    """
    Retrieve a list of valid EEG/MEG channel types from MNE library.

    This function fetches the list of acceptable channel types supported by
    the MNE library for EEG and MEG data analysis. These channel types are
    used for defining or verifying the types of channels in neurophysiological
    datasets.

    Returns:
        list of str: A list containing the names of valid EEG/MEG channel types.

    """
    return MNE_CHANNEL_TYPES.copy()

# Function to merge custom patterns with defaults
def merge_patterns(custom_patterns):
    """
    Merges custom patterns with the default patterns.
    
    Parameters
    ----------
    custom_patterns : dict
        Dictionary containing custom regex patterns to merge with defaults
        
    Returns
    -------
    dict
        Merged dictionary of patterns
    """
    patterns = DEFAULT_PATTERNS.copy()
    for category, pattern_list in custom_patterns.items():
        if category in patterns:
            patterns[category].extend(pattern_list)
        else:
            patterns[category] = pattern_list
    return patterns