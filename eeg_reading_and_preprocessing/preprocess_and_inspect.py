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
from PreprocessingFunctions import *
import multiprocessing as mp
mne.set_config('MNE_BROWSER_BACKEND', 'qt')
# if error on linux (debian) see the following link: https://web.stanford.edu/dept/cs_edu/resources/qt/install-linux
# if error on linux (arch) install qtcreator from pacman: pacman -S qtcreator
mne.set_log_level(verbose='WARNING')

#%% Load the eeg data
sampleSignalPath = '../example/eeg/PN001-original.edf' #'/path/to/edf/sample.edf'
out_path = sampleSignalPath.replace('original.edf', 'preprocessed-raw.fif')
raw = mne.io.read_raw(sampleSignalPath, preload=True)
raw = raw.pick_types(eeg=True) #focus on eeg channels (includes all channels marked as eeg not only electrode channels)


#%% plot the loaded eeg file as a seperate process so It can be kept open while continuing the analysis of the metrics
def eeg_plot(raw):
    raw.plot(duration=20, remove_dc=False, block=True, show_options=True, title='Raw EEG without any processing')
def mp_eeg_plot(raw):
    plot = (mp.Process(target=eeg_plot, args=(raw,)))
    plot.start()
    return plot
current_eeg_plot = mp_eeg_plot(raw)
# Preprocessing the file to see what basic filtering and resampling will do
#%% set parameters
l_freq = 0.5 #if should not be changed use raw.info['highpass']
h_freq = 35 #if should not be changed use raw.info['lowpass']
sfreq = 200 #if should not be changed use raw.info['sfreq']
#%% filter
raw.filter(l_freq=l_freq,h_freq=h_freq)
#%% resample
raw.resample(sfreq)

#%% now we replot the data to see the visual differences
if current_eeg_plot.is_alive():
    current_eeg_plot.join()
current_eeg_plot = mp_eeg_plot(raw)

#%% also the info of the file will be changed when we apply operations like filtering and resampling
print(f'Due to the processing of the raw object also the info of the file is changed now: \n {raw.info}')

#%% Next to the basic mne functionality the custom PreprocessingFunctions includes things like changing the montage of the signal
raw_remontaged = raw.copy()
raw_remontaged = change_montage(raw=raw_remontaged.pick(picks='eeg', exclude='bads'), montage='doublebanana')
if current_eeg_plot.is_alive():
    current_eeg_plot.join()
current_eeg_plot = mp_eeg_plot(raw_remontaged)

#%% Lastly we will save our preprocessed eeg for further processing
'''
THe main reason we decide to preprocess the eeg and keep it for further processing is that:
1. It is always good to take a look at the data one uses. Things like artifacts, bad channels etc. can cause problems.
2. By only working with the filtered and downsampled version from here on out we can speed up some processing of the data.
'''
raw.save(out_path, overwrite=False)