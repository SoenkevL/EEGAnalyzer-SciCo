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
from eeg_reading_and_preprocessing.PreprocessingFunctions import *
import multiprocessing as mp
mne.set_config('MNE_BROWSER_BACKEND', 'qt')
# if error on linux (debian) see the following link: https://web.stanford.edu/dept/cs_edu/resources/qt/install-linux
# if error on linux (arch) install qtcreator from pacman: pacman -S qtcreator
mne.set_log_level(verbose='WARNING')

#%% Load the eeg data
sampleSignalPath = 'example/eeg/PN001-original.edf' #'/path/to/edf/sample.edf'
out_path = sampleSignalPath.replace('original.edf', 'preprocessed-raw.fif')
raw = mne.io.read_raw(sampleSignalPath, preload=True)
raw = raw.pick_types(eeg=True) #focus on eeg channels (includes all channels marked as eeg not only electrode channels)


#%% plot the loaded eeg file as a seperate process so It can be kept open while continuing the analysis of the metrics
def eeg_plot(raw):
    raw.plot(duration=20, remove_dc=False, block=True, show_options=True, title='Raw EEG without any processing',
             bgcolor='w')
def mp_eeg_plot(raw):
    plot = (mp.Process(target=eeg_plot, args=(raw,)))
    plot.start()
    return plot
current_eeg_plot = mp_eeg_plot(raw)
# Preprocessing the file to see what basic filtering and resampling will do
#%% set parameters
l_freq = 1 #if should not be changed use raw.info['highpass']
h_freq = 40 #if should not be changed use raw.info['lowpass']
#%% filter
raw.filter(l_freq=l_freq,h_freq=h_freq)
#%% resample
raw.resample(raw.info['sfreq'])

#%% now we replot the data to see the visual differences
if current_eeg_plot.is_alive():
    current_eeg_plot.join()
current_eeg_plot = mp_eeg_plot(raw)

#%% also the info of the file will be changed when we apply operations like filtering and resampling
print(f'Due to the processing of the raw object also the info of the file is changed now: \n {raw.info}')

#%% We will now make use of ICA in order to identify possible noise components like eyeblinks and heartbeats within our data
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
#focusing on first 60 seconds for computational efficiency (should be extended to full file when doing actual preprocessing_
regexp = r'(EEG *)'
channel_picks = mne.pick_channels_regexp(raw.ch_names, regexp)
raw_crop = raw.copy().crop(tmax=70, tmin=10).pick(channel_picks)

#if we have an eog channel we can automatically look for eog artifacts
try:
    eog_evoked = create_eog_epochs(raw_crop).average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))
    eog_evoked.plot_joint()
except RuntimeError as e:
   print(f'Could not automatically find eog channels because: {e}')
# fit the ica components
ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(raw_crop)
print(ica)
#%%
explained_var_ratio = ica.get_explained_variance_ratio(raw_crop)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")
#%%
def plot_ica_sources(raw):
    ica.plot_sources(raw, block=True)
def mp_ica_plot(raw):
    plot = (mp.Process(target=plot_ica_sources, args=(raw,)))
    plot.start()
    return plot
current_ica_plot = mp_ica_plot(raw_crop)
#%%
ica.plot_components()
#%%
ica.plot_overlay(raw_crop, exclude=[0,1,2], picks='eeg')
#%%
ica.plot_properties(raw, picks=[0, 1])
#%%
ica.exclude=[0,1,2]
#%%
# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw_crop.copy()
ica.apply(reconst_raw)
reconst_plot = mp_eeg_plot(reconst_raw)
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