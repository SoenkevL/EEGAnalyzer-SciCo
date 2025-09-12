import mne
import time

import logging
logger = logging.getLogger(f"{__name__}")

#To ensure that all eegs are the same these should always be set, if they are equal to the files parameters they
#wont have an effect

def preprocess_eeg(eeg: mne.io.Raw, name, **kwargs) -> mne.io.Raw:
    if name =='general':
        t_start = time.time()
        result = general_preprocessing(eeg, **kwargs)
        t_elapsed = time.time() - t_start
        logger.debug(f'Preprocessing eeg using general preprocessing took {t_elapsed} seconds')
        return result
    if name =='no_additional_preprocessing':
        logger.info(f'No preprocessing applied')
        return eeg.pick_types(eeg=True)
    if name == 'paper_based':
        result = paper_based_preprocessing(eeg, **kwargs)
        return result
    logger.error(f'Preprocessing name not found')
    raise ValueError(f'Preprocessing name {name} not found')

def general_preprocessing(eeg, **kwargs):

    logger.info(f'Preprocessing eeg using general preprocessing')
    eeg = eeg.pick_types(eeg=True)
    # eeg = eeg.set_eeg_reference(kwargs.get('reference'))
    eeg = eeg.resample(kwargs.get('sfreq'))
    eeg = apply_filter(eeg, l_freq=kwargs.get('l_freq'), h_freq=kwargs.get('h_freq'))
    return eeg

def paper_based_preprocessing(eeg, **kwargs):

    logger.info(f'Preprocessing eeg using general preprocessing')
    eeg = eeg.pick_types(eeg=True)
    # eeg = eeg.set_eeg_reference(kwargs.get('reference'))
    eeg = eeg.resample(kwargs.get('sfreq'))
    eeg = apply_filter(eeg, l_freq=kwargs.get('l_freq'), h_freq=kwargs.get('h_freq'))
    eeg = eeg.notch_filter(kwargs.get('notch_freqs'))
    return eeg

def apply_filter(raw, l_freq=None, h_freq=None):
    '''
    ensures data is not filtered twice if allready at the right frequency band
    '''
    raw_h_freq_cutoff = raw.info.get('lowpass')
    raw_l_freq_cutoff = raw.info.get('highpass')
    H = h_freq if h_freq else raw_h_freq_cutoff
    L = l_freq if l_freq else raw_l_freq_cutoff
    logger.info(f'raw h freq cutoff: {raw_h_freq_cutoff}, raw l freq cutoff: {raw_l_freq_cutoff}')
    logger.info(f'new h freq cutoff: {h_freq}, new l freq cutoff: {l_freq}')
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
