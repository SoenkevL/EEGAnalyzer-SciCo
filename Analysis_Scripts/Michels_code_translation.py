import numpy as np
from scipy.signal import hilbert, welch, coherence
from scipy.signal.windows import hann, hamming
import scipy.signal as signal
from icecream import ic
ic.disable()


####### Functions from CalcFeatures.m ##########
def nextpow2(x): #checked #confirmed
    # finds next power of 2
    return np.ceil(np.log2(x))


def calculate_bands(fs): #checked #confirmed
    # Calculates hyperparameters needed for welch function in other functions aswell as alpha and delta band
    nfft = 2 ** (nextpow2(fs) + 1)  # resolution 0.5Hz
    window = nfft  # 2 second window --> resolution is 0.5 Hz
    noverlap = round(window / 2)
    Fxx = np.arange(0, fs / 2 + fs / nfft, fs / nfft)
    DeltaBand = (Fxx >= 1) & (Fxx < 4)
    AlphaBand = (Fxx >= 8) & (Fxx < 13)
    return window, noverlap, nfft, AlphaBand, DeltaBand


def calculate_signal_power(eeg_data): #checked #confirmed with small dev
    # Calculates the signal power which is the same as the standard deviation of the signal
    power = np.std(eeg_data, axis=1, ddof=1)
    return power


def calculate_adr(eeg_data, fs, window, noverlap, nfft, alpha_band, delta_band): #checked #confirmed with small dev
    # Calculates the alpha/delta ratio of the eeg per channel
    num_channels = eeg_data.shape[0]
    adr = np.zeros(num_channels)
    for i in range(num_channels):
        f, Pxx = welch(np.squeeze(eeg_data[i, :]), fs=fs, nperseg=window, noverlap=noverlap, nfft=nfft)
        adr[i] = np.sum(Pxx[alpha_band]) / np.sum(Pxx[delta_band])
    return adr

def coherence_EEGb(EEG, fs):
    # settings (needed to do a small alteration compared to matlabs implementation but the end result is almost the same)
    nfft = 4 * fs
    noverlap = 2*fs
    window = hann(nfft)

    # Calculate coherence between all electrodes
    EEGcohAll = []
    # for i in range(EEG.shape[0] - 1):
    #     f, Cxy = coherence(EEG[i, :], EEG[i+1:, :], fs=fs, window=window, noverlap=noverlap, nfft=nfft, axis=1)
    #     EEGcohAll.append(Cxy)
    #     ghjkl = 0
    for i in range(EEG.shape[0] - 1):
        for j in range(i + 1, EEG.shape[0]):
            f, Cxy = coherence(EEG[i, :], EEG[j, :], fs=fs, window=window, noverlap=noverlap, nfft=nfft, axis=0)
            EEGcohAll.append(Cxy)

    EEGcohAll = np.array(EEGcohAll)
    deltaBand = (f >= 1) & (f < 4)
    cohAll_delta = np.mean(EEGcohAll[: , deltaBand])

    return cohAll_delta


def calculate_cri(value): #checked #confirmed with small dev
    # Calculate Composite Regularity Index (CRI)
    score_signal_power = 1.0 / (1.0 + np.exp(-2 * (value['SignalPower'] - 2.5)))
    score_shannon_entropy = 1.0 / (1.0 + np.exp(-9 * (value['ShannonEntropy'] - 2.5)))
    score_adr = 1.0 / (1.0 + np.exp(-10 * (value['ADR'] - 0.5)))
    score_regularity = 1.0 / (1.0 + np.exp(-10 * (value['Regularity'] - 0.65)))
    score_coh_all_delta = 1.0 / (1.0 + np.exp(10 * (value['CohAll_delta'] - 0.45)))

    cri = score_signal_power * np.sum([score_shannon_entropy, score_adr, score_regularity, score_coh_all_delta], axis=0) / 4
    return cri


######## artifact_detection.m ###########
def artifact_detection(EEG, fs): #checked #confirmed but some of the parameter choices seem odd, maybe EEG is read different in matlab code
    fs = int(fs)
    lEEG = np.transpose(EEG.copy())
    # Define thresholds for artifact parameters
    thresh = {
        'AMP_abs': 500,      # Maximally allowed peak amplitude (absolute value, in uV) #when loading with mne values are in V so needs rescaling
        'AMP_rel': 5,        # Maximally allowed peak amplitude (relative to simultaneous amplitude in other channels)
        'FLT': 0.01,         # Maximally allowed fraction of flat EEG (standard deviation < 0.1 uV)
        'FRQ': 0.5           # Maximally allowed muscle artifact parameter
        # (power in 25-40 Hz frequency band divided by power in 4-12 Hz frequency band)
    }

    # A: calculate maximum amplitude parameter (AMP_abs)
    b, a = signal.butter(3, 0.5 / (0.5 * fs), 'high')
    EEGfiltered = signal.filtfilt(b, a, lEEG, axis=0, padtype='odd', padlen=3*(max(len(b), len(a))-1))  # high pass filtering
    EEGfiltered = np.abs(EEGfiltered[fs:-fs, :])      # take absolute values, exclude edges (filtering effects)
    AMP_abs = np.max(EEGfiltered, axis=0)             # determine maxima
    idx_max = np.argmax(EEGfiltered, axis=0)          # determine indices of maxima

    # Also exclude channels with maximum simultaneously with those above (absolute) threshold
    # Step 1 of 2: determine sample times with peak > absolute threshold (in one or more channels)
    exclchan_AMP_abs = np.where(AMP_abs >= thresh['AMP_abs'])[0]
    idx_peak_amp = np.array([], dtype=int)
    for cno in exclchan_AMP_abs:
        indices_chan = np.where(EEGfiltered[:, cno] >= thresh['AMP_abs'])[0]
        idx_peak_amp = np.union1d(idx_peak_amp, indices_chan)

    # Step 2 of 2: check maximum values in other channels. If simultaneous with peak > absolute threshold:
    # set AMP_abs to some value above threshold
    for channel in np.where(AMP_abs < thresh['AMP_abs'])[0]:
        if idx_max[channel] in idx_peak_amp:
            AMP_abs[channel] = thresh['AMP_abs'] * 2  # some value above threshold

    # Calculate the other artifact parameters channel-wise
    AMP_rel = np.full(lEEG.shape[1], np.nan)
    FLT = np.full(lEEG.shape[1], np.nan)
    FRQ = np.full(lEEG.shape[1], np.nan)

    # Preprocessing steps for relative amplitude parameter calculation
    EEG_meanamp = signal.lfilter(np.ones((fs))/fs, 1, EEGfiltered, axis=0)
    ampincluded = np.where(AMP_abs < thresh['AMP_abs'])[0]

    for channel in range(lEEG.shape[1]):
        # B: calculate relative amplitude parameter (AMP_rel)
        if channel in ampincluded:
            EEG_mean_row = EEG_meanamp[:, channel].copy()
            EEG_mean_row[EEGfiltered[:, channel] < 10] = np.nan
            AMP_rel[channel] = np.nanmax(EEG_mean_row / np.mean(EEG_meanamp[:, ampincluded[ampincluded != channel]], axis=1))

        # C: calculate flat channel parameter (FLT)
        EEGchannel = lEEG[:, channel]
        nsec = len(EEGchannel) // fs
        stdev = np.array([np.std(EEGchannel[(i * fs):((i + 1) * fs)], ddof=1) for i in range(nsec)])
        FLT[channel] = np.mean(stdev < 0.1)

        # D: calculate muscle artifact parameter (FRQ)
        F, Pxx = signal.welch(EEGchannel, fs=fs, window=hamming(10*fs), noverlap=5*fs, axis=0, nfft=2**nextpow2(10*fs))


        high_range = (F > 25) & (F <= 40)
        low_range = (F >= 4) & (F <= 12)
        FRQ_abs = np.sum(Pxx[high_range]) * (F[1] - F[0])
        if FRQ_abs > 1:
            FRQ[channel] = (np.sum(Pxx[high_range]) / np.sum(high_range)) / (np.sum(Pxx[low_range]) / np.sum(low_range))
        else:
            FRQ[channel] = 0

    # Define output variables
    artinfo = {
        'AMP_abs': AMP_abs,
        'AMP_rel': AMP_rel,
        'FLT': FLT,
        'FRQ': FRQ
    }

    excl_channel = (AMP_abs > thresh['AMP_abs']) | \
                   (AMP_rel > thresh['AMP_rel']) | \
                   (FLT > thresh['FLT']) | \
                   (FRQ > thresh['FRQ'])

    return artinfo, excl_channel


###### Regularity.m ######
def regularity(fs, EEG): #checked #confirmed with small dev
    # Squaring
    lEEG = np.transpose(EEG.copy())
    x1 = np.sqrt(lEEG ** 2)

    # Moving Window Integration
    window = int(0.5 * fs)  # Make impulse response, window in samples
    h = np.ones(window) / window
    x2 = np.apply_along_axis(lambda m: np.convolve(m, h, mode='same'), axis=0, arr=x1)

    REG = np.zeros(x2.shape[1])

    for ch in range(x2.shape[1]):
        g = x2[:, ch]  # nonnegative array g
        Q = np.flip(np.sort(g))
        N = len(Q)
        REG[ch] = np.sqrt(np.sum((np.arange(1, N + 1) ** 2) * Q) / (np.sum(Q) * (N ** 2 / 3)))  # in [0,1]

    return REG


###### transients.m ######
def detect_transients(EEG, fs): #quite significant numerical differences compared to matlab
    lEEG = EEG[10, :].copy()  # TODO: change to averaging over all channels
    L = int(fs / 2)  # Take a part of the signal (0.5 s)
    BI = 0
    partial_length = int(len(lEEG)-L)
    amp = np.zeros(partial_length)
    damp = np.zeros(partial_length)

    # Calculate amplitude of signal using Hilbert transform
    for i in range(partial_length):
        H = hilbert(lEEG[i:i + L])
        amp[i] = np.max(np.mean(np.abs(H)) - 5, 0)  # if mean amplitude < 5 uV set to zero

    # Calculate difference in amplitude
    for i in range(L, partial_length):
        damp[i] = np.max([(amp[i] - amp[i + 1 - L]), 0.1])

    # Find burst
    HH = damp > 5  # Amplitude difference has to be at least 5 uV
    HH2 = np.diff(HH.astype(int))
    burstindex = np.where(HH2 == -1)[0] - L
    burstindex = np.clip(burstindex, 0, None)  # Ensure no negative indices

    # Determine number of bursts and calculate bursts per minute
    nburst = len(burstindex)
    bursts_per_minute = nburst / (len(lEEG) / (fs * 60))
    nburst = np.min([nburst, 100])  # Maximize number of calculations

    if nburst > 1:
        BI = np.zeros((nburst**2 - nburst) // 2)
        bursts = np.zeros((nburst, round(1.1 * fs)))

        # Put all bursts (from -0.1 to 1 s) in a matrix
        for kk in range(nburst):
            if burstindex[kk] - round(fs / 10) > 0:
                BB = lEEG[burstindex[kk] - round(fs / 10):burstindex[kk] + fs]
            else:
                BB = np.concatenate([np.zeros(round(fs / 10)), lEEG[burstindex[kk]:burstindex[kk] + fs]])
            bursts[kk, :] = BB

        # Calculate correlations between bursts
        ccounter = 0
        for aa in range(nburst):
            for bb in range(aa + 1, nburst):
                b1 = bursts[aa, :fs]
                b2 = bursts[bb, :fs]
                BI[ccounter] = np.max(np.true_divide(np.correlate(b1, b2, mode='full'),
                                                     np.sqrt(np.dot(b1, b1) * np.dot(b2, b2))))
                ccounter += 1

    return bursts_per_minute, BI


##### ShannonEntropy.m #####
def shannon_entropy(eeg_data):
    eeg_data = np.transpose(eeg_data.copy())
    _, M = eeg_data.shape  # Number of EEG channels
    ent_value = np.zeros(M)
    x = np.arange(-200, 201)  # Define range for histogram

    for chan in range(M):
        n, _ = np.histogram(eeg_data[:, chan], bins=x)
        np_hist = n[n > 0] / np.sum(n)
        ent_value[chan] = -np.sum(np_hist * np.log2(np_hist))
    return ent_value


######## Functions from calculate_qeeg_features.m #########
def calculate_qeeg_features(EEG, fs):
    fs = int(fs)
    # Settings
    cutoff_value = 10  # Maximum amplitude of "suppression" (microvolts)
    min_supp_duration = 0.5  # Minimum duration of "suppression" (seconds)
    mininterval = 0.2  # Minimum duration of "bursts" (seconds)
    bsar_lim = [0.01, 0.99]  # Upper and lower bounds of BCI for which BSAR is calculated
    minchannels = 12

    # Detect suppressions
    ic(f'cqf {EEG.shape}')
    _, art_channels = artifact_detection(EEG, fs)
    Artifacts = np.sum(art_channels)

    if len(art_channels)-np.sum(art_channels) >= minchannels:
        # 1. Remove channels that were rejected
        lEEG = np.transpose(EEG.copy())
        lEEG = lEEG[:, ~art_channels]
        EEG_bin = np.abs(lEEG) < (cutoff_value / 2)
        EEG_supp = np.zeros_like(lEEG)
        for channel in range(lEEG.shape[1]):
            # Only keep "suppressions" with minimum duration
            signal = np.concatenate([[0], EEG_bin[:, channel], [0]])
            ii1 = np.where(np.diff(signal) == 1)[0]
            ii2 = np.where(np.diff(signal) == -1)[0] - 1
            ii = (ii2 - ii1 + 1) >= round(min_supp_duration * fs)
            ii1 = ii1[ii]
            ii2 = ii2[ii]
            for idx in range(len(ii1)):
                EEG_supp[ii1[idx]:ii2[idx] + 1, channel] = 1

            # Remove "bursts" shorter than minimum duration
            signal = np.concatenate([[1], EEG_supp[:, channel], [1]])
            ii1 = np.where(np.diff(signal) == -1)[0]
            ii2 = np.where(np.diff(signal) == 1)[0] - 1
            ii = (ii2 - ii1 + 1) < round(mininterval * fs)
            ii1 = ii1[ii]
            ii2 = ii2[ii]
            for idx in range(len(ii1)):
                EEG_supp[ii1[idx]:ii2[idx] + 1, channel] = 1

        # Remove edges of EEG (filtering/windowing effects)
        lEEG = lEEG[fs:-fs, :]
        EEG_supp = EEG_supp[fs:-fs, :]
        supp_sum = np.sum(EEG_supp)

        # Calculate qEEG parameters (per channel)
        BCI_chan = np.nan * np.zeros(lEEG.shape[1])
        BSAR_chan = np.nan * np.zeros(lEEG.shape[1])

        for channel in range(lEEG.shape[1]):
            # Calculate BCI
            BCI_chan[channel] = 1 - np.mean(EEG_supp[:, channel])

            # Calculate BSAR
            if bsar_lim[0] <= BCI_chan[channel] <= bsar_lim[1]:
                powburst = np.std(lEEG[~EEG_supp[:, channel].astype(bool), channel], ddof=1)
                powsupp = np.std(lEEG[EEG_supp[:, channel].astype(bool), channel], ddof=1)
                BSAR_chan[channel] = powburst / powsupp
            else:
                BSAR_chan[channel] = 1

        # Calculate and save final results
        BCI = np.mean(BCI_chan)
        BSAR = np.mean(BSAR_chan)
    else:
        BCI = None
        BSAR = None

    return Artifacts, BCI, BSAR

def crosscorr(b1, b2):
    b1 = b1 - np.mean(b1)
    b2 = b2 - np.mean(b2)
    corr = np.true_divide(np.correlate(b1, b2, mode='full'),
                          np.sqrt(np.dot(b1, b1) * np.dot(b2, b2)))
    lags = signal.correlation_lags(len(b1), len(b2), mode='full')
    correct_idxs = np.where(np.abs(lags) < min(len(b1), len(b2), 21))
    correct_lags = lags[correct_idxs]
    coorect_corr = corr[correct_idxs]
    coorect_corr = coorect_corr[::-1]
    return correct_lags, coorect_corr


def gpd_analysis(EEG, fs):
    lEEG = np.transpose(EEG.copy())
    fs = int(fs)
    # I. Settings
    ma_window = 120                  # window of moving average filter in milliseconds
    min_peak_distance = 200 / 1000   # Minimum distance between spikes in seconds
    min_peak_duration = 60 / 1000    # Minimum peak duration in seconds (to classify as GPD)
    max_peak_duration = 500 / 1000   # Maximum peak duration in seconds (to classify as GPD; otherwise classified as burst)
    min_peak_amplitude = 2 * 20     # Minimum difference between minimum and maximum value of discharge segment in uV
    treshold_epoch_length = 5        # Length of (overlapping) EEG epochs in seconds for which treshold is calculated
    treshold_constant = 0.6          # Constant used in tresholding --> higher constant will select less peaks
    percentile = 75                  # Parameter used in tresholding --> higher percentile will select less peaks
    num_segments_comp = 10           # number of GPDs to compare in cross correlation analysis
    periodicity_interval = 0.25      # bandwidth parameter in periodicity analysis

    # filtFreq = [0.5, 25]             # frequency band for Butterworth bandpass filter
    # filtOrder = 3                    # order of Butterworth bandpass filter
    # b, a = signal.butter(filtOrder, np.array(filtFreq) / (0.5 * fs), btype='band', output='ba')
    min_peak_channels=np.floor(0.6*lEEG.shape[1])

    # III. GPD detection algorithm (consisting of 3 filtering steps)
    peak_vector = np.zeros_like(lEEG)

    for chan in range(lEEG.shape[1]):

        # Filter 1: NLEO (NonLinear Energy Operator)
        EEG_nleo = np.zeros_like(lEEG)
        for x in range(3, lEEG.shape[0]):
            EEG_nleo[x, chan] = abs((lEEG[x-1, chan] * lEEG[x-2, chan]) - (lEEG[x, chan] * lEEG[x-3, chan]))

        # Filter 2: moving average filter
        EEG_ma = np.zeros_like(lEEG)
        b = np.ones(int(np.ceil(ma_window * (fs / 1000)))) / int(np.ceil(ma_window * (fs / 1000)))
        a = 1
        EEG_ma[:, chan] = signal.filtfilt(b, a, EEG_nleo[:, chan])

        # Filter 3: adaptive tresholding
        EEG_treshold = np.zeros_like(lEEG)
        epoch_length = treshold_epoch_length * fs
        step_size = fs
        steps = range(0, EEG_ma.shape[0] - epoch_length, step_size)

        #EEG_ma seems to be very accurate when comparing it with python
        for epoch_startpoint in steps:
            epoch_endpoint = epoch_startpoint + epoch_length if epoch_startpoint != steps[-1] else EEG_ma.shape[0]
            epoch = EEG_ma[epoch_startpoint:epoch_endpoint, chan]
            tperc = np.percentile(epoch, percentile)
            tstd = np.std(epoch, ddof=1)
            treshold = treshold_constant * (tstd + tperc)

            for j in range(epoch_startpoint, epoch_endpoint):
                EEG_treshold[j, chan] = 1 if EEG_ma[j, chan] >= treshold else 0
            stopper = 0

        # Step 4: additional operations
        EEG_treshold[0, chan] = EEG_treshold[-1, chan] = 0
        start_segment = np.where(np.diff(EEG_treshold[:, chan]) > 0)[0] + 1
        end_segment = np.where(np.diff(EEG_treshold[:, chan]) < 0)[0]
        segment_vector = np.vstack((start_segment, end_segment)).T

        if segment_vector.size > 0:
            # a. Remove segments that are too short or too long
            diff_vector = segment_vector[:, 1] - segment_vector[:, 0]
            valid_segments = (min_peak_duration * fs <= diff_vector) & (diff_vector < max_peak_duration * fs)
            segment_vector_orig = segment_vector
            segment_vector = segment_vector[valid_segments]

            # b. Remove segments whose amplitude is too low
            amplitude_cutoff = [i for i in range(segment_vector.shape[0])
                                if (np.max(lEEG[segment_vector[i, 0]:segment_vector[i, 1], chan]) -
                                    np.min(lEEG[segment_vector[i, 0]:segment_vector[i, 1], chan])) > min_peak_amplitude]
            segment_vector_orig2 = segment_vector
            segment_vector = segment_vector[amplitude_cutoff]

            # c. Create a peak_vector to highlight peaks
            for i in range(segment_vector.shape[0]):
                peak_vector[segment_vector[i, 0]:segment_vector[i, 1], chan] = 1

    # IV. GPD analysis (after detection algorithm)
    results = {}
    #checked up untill here but there may still be small differences in the peak extraction I checked the first 3 eeg channels so far
    # 1. Calculate generalized (periodic) discharge frequency
    peak_sum_vector = np.sum(peak_vector, axis=1)
    gd_locs, _ = signal.find_peaks(peak_sum_vector, height=min_peak_channels, distance=int(min_peak_distance * fs)) # makes sense as we are looking for a number of channels having a peak at the same time
    gd_intervals = np.diff(gd_locs)
    gpd_freq_median = 1 / (np.median(gd_intervals) / fs) if len(gd_intervals) > 0 else 0
    results['RPP'] = gpd_freq_median if gpd_freq_median > 0 else 0

    # 2. Calculate relative power of generalized discharges
    peak_vector_generalized = peak_vector.copy()
    peak_vector_generalized[0, :] = peak_vector_generalized[-1, :] = 0

    start_and_stop_vectors = []
    for channel in range(peak_vector.shape[1]):
        start_vector = np.where(np.diff(peak_vector_generalized[:, channel]) > 0)[0] + 1
        stop_vector = np.where(np.diff(peak_vector_generalized[:, channel]) < 0)[0] + 1 #changed from original
        start_and_stop_vectors.append((start_vector, stop_vector))

        for j in range(len(start_vector)):
            if np.max(peak_sum_vector[start_vector[j]:stop_vector[j]]) <= min_peak_channels:
                peak_vector_generalized[start_vector[j]:stop_vector[j], channel] = 0

    gpd_power = np.sum(lEEG[peak_vector_generalized == 1] ** 2)
    signal_power = np.sum(lEEG ** 2)
    results['RPPpower'] = gpd_power / signal_power

    # 3. Calculate periodicity
    if results['RPP'] > 0.2:
        periodicity = 1 - (np.sum(gd_intervals < (np.median(gd_intervals) * (1 - periodicity_interval))) +
                           np.sum(gd_intervals > (np.median(gd_intervals) * (1 + periodicity_interval)))) / len(gd_intervals)
        results['Periodicity'] = periodicity
    else:
        results['Periodicity'] = 0

    # 4. Calculate correlation of generalized discharges
    if results['RPP'] > 0.2:
        mean_crosscorrelation_chan = np.zeros(lEEG.shape[1])
        for channel in range(lEEG.shape[1]):
            if np.sum(peak_vector_generalized[:, channel]) > 0:
                start_peaks, stop_peaks = start_and_stop_vectors[channel]
                corr_channel = np.zeros((len(start_peaks) - num_segments_comp, num_segments_comp))

                for gen_segment in range(num_segments_comp, len(start_peaks)):
                    segment1_gen = lEEG[start_peaks[gen_segment]:stop_peaks[gen_segment], channel]
                    for gen_segment_ref in np.arange(1, num_segments_comp, 1):
                        segment2_gen = lEEG[start_peaks[gen_segment - gen_segment_ref]:stop_peaks[gen_segment - gen_segment_ref], channel]
                        lags, temp_corr = crosscorr(segment1_gen, segment2_gen)
                        corr_channel[gen_segment - num_segments_comp, gen_segment_ref] = np.max(temp_corr) #hard to compare to origina due to the difference in peak finding

                mean_crosscorrelation_chan[channel] = np.nanmean(corr_channel)
            else:
                mean_crosscorrelation_chan[channel] = np.nan
        results['Burstc'] = np.nanmean(mean_crosscorrelation_chan)
    else:
        results['Burstc'] = 0

    '''
    correlation and periodicity have differences which are quite significant.
    I think its mainly due to the propagation of rounding errors/numerical differences to matlab in the script.
    This means that a relative comparison might be possible using only metrics calced with python.
    Comparing matlab to python results would be hard as for that the numericla precision would need to be unified
    '''

    # 5. Calculate background continuity
    artifacts, continuity, BSAR = calculate_qeeg_features(EEG, fs)
    results.update({'BCI': continuity, 'BSAR': BSAR, 'artefacts': artifacts})

    # Calculate continuity_corrected
    if results['BCI']:
        BCI_c = results['BCI'] - np.mean(peak_vector)
        results['BCIc'] = BCI_c
    else:
        results['BCIc'] = None

    return results


def calc_features_from_EEG_data(EEG_data, fs, interval_len=10):
    EEG_data = np.transpose(EEG_data)
    no_chan, no_datapoint = EEG_data.shape
    interval_len_samples = interval_len*fs
    #initialize metrics to be calculated
    artifact_counter = []
    ADR = []
    SignalPower = []
    Shannon_Entropy = []
    CohAll_delta = []

    #calculate metrics per 10 sec epoch
    for idx in np.arange(0, no_datapoint-interval_len_samples, interval_len_samples):
        EEG_data_int = EEG_data[:, idx:idx+interval_len_samples]
        #Feature calc
        ADR.append(calculate_adr(EEG_data_int, fs, *calculate_bands(fs)))  # ADR checked returns the same array
        artinfo, artchannels = artifact_detection(EEG_data_int, fs)  # checked returns same info and channels (maybe needs another check as there is still a small discrepancy)
        artifact_counter.append(np.sum(artchannels))
        SignalPower.append(calculate_signal_power(EEG_data_int))  # checked returns the same power
        Shannon_Entropy.append(shannon_entropy(EEG_data_int))  # checked returns same entropy value
        CohAll_delta.append(coherence_EEGb(EEG_data_int, fs))  # checked, could not implement the original version of the function due to the parameter settings not beeing allowed in python (they are a bit odd in the original function I have to say)

    #calculate means of 10 sec epochs and channels
    ADR_mean = np.mean(ADR)
    SignalPower_mean = np.mean(SignalPower)
    Shannon_Entropy_mean = np.mean(Shannon_Entropy)
    CohAll_delta_mean = np.mean(CohAll_delta)
    artifact_counter_mean = np.mean(artifact_counter)

    #calculate regularity over 5min epoch
    reg = regularity(fs, EEG_data) #checked, small numerical deviations
    reg_mean = np.mean(reg)

    #put metrics calculated so far into struct
    value = {
        'ADR': ADR_mean,
        'SignalPower': SignalPower_mean,
        'ShannonEntropy': Shannon_Entropy_mean,
        'CohAll_delta': CohAll_delta_mean,
        'Artifact': artifact_counter_mean,
        'Regularity': reg_mean
    }

    #calculate cri from values so far
    cri = calculate_cri(value) #checked, small numerical deviations

    #transients
    bpm, BI = detect_transients(EEG_data, fs) #checked, small numerical differences can lead to small deviations in results

    transient_values = {
        'bursts_per_minute': bpm,
        'mean_burst_correlations': np.mean(BI),
        'max_bursts_correlations': np.max(BI),
        'fractions_bursts_corr_larger_08': np.sum(BI>0.8)/len(BI) if not isinstance(BI, int) else BI
    }

    results = gpd_analysis(EEG_data, fs)

    final_values = {**value, 'CRI': cri, **transient_values, **results}
    final_values = dict(sorted(final_values.items()))
    return final_values
