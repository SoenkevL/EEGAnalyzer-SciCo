'''
# Preprocessing steps for an EEG
- Load the data
- Print information about the data
- check the file visually ('bad' annotations can be added here for artifacts)
- filter between 0.5 and 35 hz
- resample to 100hz sampling freq
- recheck psd and data
- save file to fif format
'''

#%% imports and settings
import mne
mne.set_log_level(verbose='WARNING')

#%% Load the eeg data
sampleSignalPath = 'example/eeg/PN001-original.edf' #'/path/to/edf/sample.edf'
out_path = sampleSignalPath.replace('original.edf', 'preprocessed-raw.fif')
raw = mne.io.read_raw(sampleSignalPath, preload=True)
raw = raw.pick_types(eeg=True) #focus on eeg channels (includes all channels marked as eeg not only electrode channels)

# Preprocessing the file to see what basic filtering and resampling will do
#%% set parameters
l_freq = 0.5 #if should not be changed use raw.info['highpass']
h_freq = 35 #if should not be changed use raw.info['lowpass']
sfreq = 200 #if should not be changed use raw.info['sfreq']
#%% filter
raw.filter(l_freq=l_freq,h_freq=h_freq)
#%% resample
raw.resample(sfreq)

#%% Lastly we will save our preprocessed eeg for further processing
'''
THe main reason we decide to preprocess the eeg and keep it for further processing is that:
1. It is always good to take a look at the data one uses. Things like artifacts, bad channels etc. can cause problems.
2. By only working with the filtered and downsampled version from here on out we can speed up some processing of the data.
'''
raw.save(out_path, overwrite=False)