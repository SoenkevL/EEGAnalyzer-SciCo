'''
Here I write custom function for the reproduction of the paper.
These are based on functions provided in the repository https://github.com/BIAPT/Criticality_PCI_Anesthesia
'''
import neurokit2 as nk
import numpy as np
import edgeofpy as eop
import pandas as pd

# Features Pred
def custom_mse(channel_data):
    dims, _ = nk.complexity_dimension(channel_data)
    mse, _ = nk.entropy_multiscale(channel_data, show=False, dimension=dims)
    return mse

# Avalanche features
def custom_avalanche_functions(data):
    '''
    Code is mostly copied from the afformentioned repository and slightly modified to work with my pipeline
    data needs to be a 2 dim array with the first dimension being the channels and the second the time points
    '''
    # output
    out = {'mean_iei': [],
           'tau': [],
           'tau_dist': [],
           'tau_dist_TR': [],
           'alpha': [],
           'alpha_dist': [],
           'alpha_dist_TR': [],
           'third': [],
           'dcc_cn': [],
           'avl_br': [],
           'br': [],
           'rep_dissimilarity_avg': [],
           'rep_size': [],
           'fano': [],
           'chi_test': [],
           'chi_notest': [],
           'sig_length': [],
           'len_avls': [],
           'data_mean': [],
           'data_std': []
           }
    fs = 1450
    sig_length = min(data.shape[1] / fs, 300)
    nr_channels = data.shape[0]
    THRESH_TYPE = 'both' # Fosque22: 'both'

    GAMMA_EXPONENT_RANGE = (0, 2)
    LATTICE_SEARCH_STEP = 0.1

    # read out arguments
    BIN_THRESHOLD = 3
    MAX_IEI = 0.04
    BRANCHING_RATIO_TIME_BIN = 0.04
    data_mean = np.mean(np.abs(data))
    data_std = np.std(data)

    events_by_chan = eop.binarized_events(data, threshold=BIN_THRESHOLD,
                                thresh_type=THRESH_TYPE, null_value=0)
    events_one_chan = np.sum(events_by_chan, axis=0)


    #################################
    #    Avalanches                 #
    #################################

    # Detect avalanches
    #breakpoint()
    avls, _, _, mean_iei = eop.detect_avalanches(events_by_chan, fs,
                                                 max_iei=MAX_IEI,
                                                 threshold=BIN_THRESHOLD,
                                                 thresh_type=THRESH_TYPE)

    sizes = [x['size'] for x in avls]
    dur_bin = [x['dur_bin'] for x in avls]
    dur_sec = [x['dur_sec'] for x in avls]
    len_avls = len(avls)

    #################################
    #    TAU                 #
    #################################
    # Estimate fit and extract exponents with min and max of data

    size_fit = eop.fit_powerlaw(sizes, xmin=1, discrete = True, xmax = None)
    tau = size_fit['power_law_exp']
    tau_dist = size_fit['best_fit']
    tau_dist_TR = size_fit['T_R_sum']


    #################################
    #    ALPHA                     #
    #################################

    #dur_bin_fit = eop.fit_powerlaw(dur_bin, discrete = True)
    #alpha_bin = dur_bin_fit['power_law_exp']

    dur_fit = eop.fit_powerlaw(dur_sec, xmin='min', xmax = None, discrete = False)
    alpha = dur_fit['power_law_exp']
    alpha_dist = dur_fit['best_fit']
    alpha_dist_TR = dur_fit['T_R_sum']


    #################################
    #    Third   and DCC            #
    #################################

    #third_bin = eop.fit_third_exponent(sizes, dur_bin, discrete= True)
    third = eop.fit_third_exponent(sizes, dur_sec, discrete= False, method = 'pl')

    #dcc_cn_bin = eop.dcc(tau, alpha_bin, third_bin)
    dcc_cn = eop.dcc(tau, alpha, third)


    #################################
    #    REPERTPOIRE               #
    #################################

    # Estimate avalanche functional repertoire
    repertoire = eop.avl_repertoire(avls)
    # normalize the repertoire by signal length
    rep_size = repertoire.shape[0]/sig_length

    #################################
    #    Branching Ratio            #
    #################################
    # Calculate avalanche branching ratio
    avl_br = eop.avl_branching_ratio(avls)
    # Calculate branching ratio
    br = eop.branching_ratio(events_one_chan, BRANCHING_RATIO_TIME_BIN, fs)


    #################################
    #   Susceptibility              #
    #################################

    # Calculate Fano factor
    fano = eop.fano_factor(events_one_chan)

    # Calculate susceptibility
    chi_test, _ = eop.susceptibility(events_by_chan,test = True)
    chi_notest, _ = eop.susceptibility(events_by_chan,test = False)

    ## Save output
    for name in out.keys():
        out[name].append(locals()[name])

    output_df = pd.DataFrame(out)