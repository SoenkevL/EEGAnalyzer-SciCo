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
It showcases how metrics can be added for the analysis.
The most important thing is that this file has a calculate function that returns the result of the metric calculation
as a dictionary name: value

Please feel free to use this file as a template for your own metrics.
"""

import edgeofpy as eop
import neurokit2 as nk
import numpy as np
import logging
import time

PER_CHANNEL = True

####mandatory function to choose a metric set in the pipeline####
def calculate(data, name, **kwargs) -> dict[str, float]:
    logger = logging.getLogger(f"{__name__}")
    result=None
    t_start = time.time()
    if name =='lzc_only':
        logger.debug(f'Calculating lzc for data')
        result = lzc_adapted(data)
    t_elapsed = time.time() - t_start
    logger.debug(f'Calculating {name} took {t_elapsed} seconds')
    return result

#### custom additional functions ####
def lzc_adapted(channel_input: np.ndarray) -> dict:
    lzc, info = nk.complexity_lempelziv(channel_input)
    return {'lzc': lzc}

# U are encouraged to add more metrics here and adapt the calculate function to use them


