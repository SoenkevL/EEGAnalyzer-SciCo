import mne


#Mandatory parameters, will be saved alongside results as well
#To ensure that all eegs are the same these should always be set, if they are equal to the files parameters they
#wont have an effect
SFREQ = 500
H_FREQ_CUTOFF = 35
L_FREQ_CUTOFF = 1
REFERENCE = 'average'

def preprocess_eeg(eeg: mne.io.Raw) -> mne.io.Raw:
    current_reference = eeg.info['chs'][0]['loc'][6]
    eeg = eeg.pick_types(eeg=True)
    eeg = eeg.set_eeg_reference(REFERENCE)
    eeg = eeg.resample(SFREQ)
    eeg = eeg.filter(L_FREQ_CUTOFF, H_FREQ_CUTOFF)
    return eeg
