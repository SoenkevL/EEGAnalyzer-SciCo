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
import logging
import numpy as np

def lzc_adapted(channel_input: np.ndarray) -> dict:
    lzc, info = nk.complexity_lempelziv(channel_input)
    return {'lzc': lzc}

#mandatory function to choose a metric set in the pipeline
def select_metrics(name: str) -> dict:
    if name =='lzc_only':
        return {
            'metric_funcs' : [lzc_adapted],
            'channelwise' : True
        }

    #TODO: update to new metrics functions usage
    elif name =='final-metric-set-soenkes-thesis':
        return {
            'metrics_name_list' : ['fractal_dimension_katz', 'fractal_dimension_higuchi_k-10', 'fractal_dimension_hurst',
                                 'permutation_entropy', 'multiscale_entropy', 'multiscale_permutation_entropy',
                                 'lempel_ziv_complexity', 'largest_lyapunov_exponent'],
            'metrics_functions' : [nk.fractal_katz, nk.fractal_higuchi, nk.fractal_hurst,
                                 nk.entropy_permutation, nk.entropy_multiscale, nk.entropy_multiscale,
                                 nk.complexity_lempelziv, nk.complexity_lyapunov],
            'kwargs_list' : [None, {'k_max': 10}, None,
                           None, None, {'method': 'MSPEn'},
                           None, None],
            'channelwise' : True,
        }

    logging.error(f'Error in metric selection, name {name} is not a valid option')
    return None