"""
Copyright (C) <2025>  <Soenke van Loh>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

Metrics for EEG analysis.

This module provides functions for selecting and calculating metrics for EEG analysis.
This particular example was used during my Thesis (https://essay.utwente.nl/104520/).
It showcases how metrics can be added for the analysis.
The most important thing is that this file has a select_metrics function that returns the metrics functions, names and
kwargs for a given metric set. The metrics functions are then used in the analysis.py file to calculate the metrics.

Please feel free to use this file as a template for your own metrics.
"""

import edgeofpy as eop
import neurokit2 as nk
import numpy as np

PER_CHANNEL = True
METRIC_NAME = 'lzc_only'

####mandatory function to choose a metric set in the pipeline####
def calculate(data, name=METRIC_NAME):
    if name =='lzc_only':
        return lzc_adapted(data)
    return None

#### custom additional functions ####
def lzc_adapted(channel_input: np.ndarray) -> dict:
    lzc, info = nk.complexity_lempelziv(channel_input)
    return {'lzc': lzc}

def first_metric_set_MysticalEntropy(channel_input: np.ndarray) -> dict:
    """
    uses single channel input
    - lzc
    - multiscale entropy
    - spectral entropy
    - permutation entropy
    - fratal katz
    - fractal higuchi
    """
    # calculate the metrics
    lzc, _ = nk.complexity_lempelziv(channel_input)
    dimension, _ = nk.complexity_dimension(channel_input)
    multiscale_entropy, _ = nk.entropy_multiscale(channel_input, dimension=dimension)
    spectral_entropy, _ = nk.entropy_spectral(channel_input)
    fractal_dimension_katz, _ = nk.fractal_katz(channel_input)
    fractal_dimension_higuchi, _ = nk.fractal_higuchi(channel_input)

    return {
        'lzc': lzc,
        'dim' : dimension,
        'msen': multiscale_entropy,
        'spen': spectral_entropy,
        'fdk': fractal_dimension_katz,
        'fdh': fractal_dimension_higuchi,
    }

def c_pcipipe_eoc(channel_input: np.ndarray) -> dict:
    pass

def c_pcipipe_dfa(channel_input: np.ndarray) -> dict:
    pass

def c_pcipipe_avc(channel_input: np.ndarray) -> dict:
    pass

def add_metrics_mystical_entropy(channel_input: np.ndarray) -> dict:
    pass

#mandatory functions to preprocess eeg and choose a metric set in the pipeline



