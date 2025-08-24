import mne
import time

import logging
logger = logging.getLogger(f"{__name__}")

#Mandatory parameters, will be saved alongside results as well
#To ensure that all eegs are the same these should always be set, if they are equal to the files parameters they
#wont have an effect
SFREQ = 500
H_FREQ_CUTOFF = 35
L_FREQ_CUTOFF = 1
REFERENCE = 'average'


def preprocess_eeg(eeg: mne.io.Raw, name) -> mne.io.Raw:
    if name =='general':
        t_start = time.time()
        result = general_preprocessing(eeg)
        t_elapsed = time.time() - t_start
        logger.debug(f'Preprocessing eeg using general preprocessing took {t_elapsed} seconds')
        return result
    return eeg

def general_preprocessing(eeg):
    logger.info(f'Preprocessing eeg using general preprocessing')
    eeg = eeg.pick_types(eeg=True)
    # eeg = eeg.set_eeg_reference(REFERENCE)
    eeg = eeg.resample(SFREQ)
    eeg = apply_filter(eeg)
    return eeg

def apply_filter(raw):
    '''
    ensures data is not filtered twice if allready at the right frequency band
    '''
    raw_h_freq_cutoff = raw.info.get('lowpass')
    raw_l_freq_cutoff = raw.info.get('highpass')
    H = H_FREQ_CUTOFF if H_FREQ_CUTOFF else raw_h_freq_cutoff
    L = L_FREQ_CUTOFF if L_FREQ_CUTOFF else raw_l_freq_cutoff
    logger.info(f'h freq cutoff: {H_FREQ_CUTOFF}, l freq cutoff: {L_FREQ_CUTOFF}')
    logger.info(f'raw h freq cutoff: {raw_h_freq_cutoff}, raw l freq cutoff: {raw_l_freq_cutoff}')
    if raw_h_freq_cutoff > H and raw_l_freq_cutoff < L:
        logger.info(f'Applying bandpass filter to raw eeg')
        raw = raw.filter(L, H)
    elif raw_h_freq_cutoff <= H and raw_l_freq_cutoff > L:
        logger.info(f'Applying highpass filter to raw eeg')
        raw = raw.filter(L, None)
    elif raw_h_freq_cutoff > H and raw_l_freq_cutoff >= L:
        logger.info(f'Applying lowpass filter to raw eeg')
        raw = raw.filter(None, H)
    else:
        logger.info(f'No filter applied to raw eeg because the frequency band is allready within bounds')
    return raw
