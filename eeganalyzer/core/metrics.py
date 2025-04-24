"""
Metrics for EEG analysis.

This module provides functions for selecting and calculating metrics for EEG analysis.
"""

import edgeofpy as eop
import neurokit2 as nk


def select_metrics(name):
    """
    Selects a set of metrics based on the provided name.
    
    Args:
        name (str): Name of the metric set.
        
    Returns:
        tuple: (metrics_functions, metrics_name_list, kwargs_list)
            - metrics_functions (list): List of metric functions to calculate on the time series.
            - metrics_name_list (list): List of names for the functions, used to save the results.
            - kwargs_list (list): List of dictionaries with additional arguments for the functions.
    """
    # TODO update this file to have some more generally sensible categories
    if name == 'old_without_chaos':  ###################################################################################
        metrics_name_list = ['complexity_lempel_ziv', 'fractal_dimension_katz', 'entropy_multiscale',
                             'aval_fano_factor', 'entropy_shannon', 'entropy_permutation',
                             'Largest Lyapunov exponent']
        metrics_functions = [nk.complexity_lempelziv, nk.fractal_katz, nk.entropy_multiscale,
                             eop.avalanche.fano_factor, nk.entropy_shannon, nk.entropy_permutation,
                             nk.complexity_lyapunov]
        kwargs_list = [None, None, None, None, None, None, None]
        return metrics_functions, metrics_name_list, kwargs_list

    elif name == 'old_with_chaos':  ####################################################################################
        # initialize the ml engine
        try:
            import matlab.engine
            ml_engine = matlab.engine.start_matlab()
        except Exception as e:
            print(f'could not instantiate matlab engine. See error below\n {e}')
            return None, None, None
        metrics_name_list = ['complexity_lempel_ziv', 'fractal_dimension_katz', 'entropy_multiscale',
                             'aval_fano_factor', 'entropy_shannon', 'entropy_permutation',
                             'Largest Lyapunov exponent']
        metrics_functions = [nk.complexity_lempelziv, nk.fractal_katz, nk.entropy_multiscale,
                             eop.avalanche.fano_factor, nk.entropy_shannon, nk.entropy_permutation,
                             nk.complexity_lyapunov]
        kwargs_list = [None, None, None, None, None, None, None]

        # setup parameters for chaos pipeline (due to problems with kwargs and matlab engines these have to be in the right
        # order and will be set as a list of arguments instead of keyword arguments
        cfg_pipeline = {'cutoff': 0.84, 'stationarity_test': 'bvs', 'denoising_algorithm': 'schreiber',
                        'gaussian_transform': 0, 'surrogate_algorithm': 'aaft_cpp', 'downsampling_method': 'downsample',
                        'sigima': 0.5}
        metrics_name_list = metrics_name_list + ['result_cp', 'entropy_permutation_cp', 'K_cp']
        metrics_functions = metrics_functions + [ml_engine.chaos_modified]
        kwargs_list = kwargs_list + [cfg_pipeline]
        return metrics_functions, metrics_name_list, kwargs_list

    elif name == 'optimize_embeddings':  ###############################################################################
        metrics_name_list = ['delay_rosenstein_1994', 'delay_fraser_1986', 'delay_lyle_2021',
                             'embedding_cd', 'embedding_fnn', 'embedding_afn']
        metrics_functions = [nk.complexity_delay, nk.complexity_delay, nk.complexity_delay,
                             nk.complexity_dimension, nk.complexity_dimension, nk.complexity_dimension]
        kwargs_list = [{'method': 'rosenstein1994'}, {'method': 'fraser1986'}, {'method': 'lyle2021'},
                       {'method': 'cd'}, {'method': 'fnn'}, {'method': 'afn'}]
        return metrics_functions, metrics_name_list, kwargs_list

    elif name == 'optimize_embeddings_fast':  ##########################################################################
        metrics_name_list = ['delay_rosenstein_1994', 'delay_fraser_1986', 'delay_lyle_2021',
                             'embedding_fnn', 'embedding_afn']
        metrics_functions = [nk.complexity_delay, nk.complexity_delay, nk.complexity_delay,
                             nk.complexity_dimension, nk.complexity_dimension]
        kwargs_list = [{'method': 'rosenstein1994'}, {'method': 'fraser1986'}, {'method': 'lyle2021'},
                       {'method': 'fnn'}, {'method': 'afn'}]
        return metrics_functions, metrics_name_list, kwargs_list

    elif name == 'additional':  ########################################################################################
        metrics_name_list = ['fractal_dimension_hurst', 'entropy_multiscale_pe',
                             'entropy_permutation_delay-30', 'weighted_entropy_permutation']
        metrics_functions = [nk.fractal_hurst, nk.entropy_multiscale,
                             nk.entropy_permutation, nk.entropy_permutation]
        kwargs_list = [None, {'method': 'MSPEn'},
                       {'delay': 30}, {'weighted': True}]
        return metrics_functions, metrics_name_list, kwargs_list

    elif name == 'higuchi':  ########################################################################################
        metrics_name_list = ['fractal_dimension_higuchi_k-10']
        metrics_functions = [nk.fractal_higuchi]
        kwargs_list = [{'k_max': 10}]
        return metrics_functions, metrics_name_list, kwargs_list

    elif name =='final-0':
        metrics_name_list = ['fractal_dimension_katz', 'fractal_dimension_higuchi_k-10', 'fractal_dimension_hurst',
                             'permutation_entropy', 'multiscale_entropy', 'multiscale_permutation_entropy',
                             'lempel_ziv_complexity', 'largest_lyapunov_exponent']
        metrics_functions = [nk.fractal_katz, nk.fractal_higuchi, nk.fractal_hurst,
                             nk.entropy_permutation, nk.entropy_multiscale, nk.entropy_multiscale,
                             nk.complexity_lempelziv, nk.complexity_lyapunov]
        kwargs_list = [None, {'k_max': 10}, None,
                       None, None, {'method': 'MSPEn'},
                       None, None]
        return metrics_functions, metrics_name_list, kwargs_list

    elif name =='eof_0-1chaos':
        metrics_name_list = ['K-pipeline', 'K-pipeline_denoised', 'K']
        metrics_functions = [eop.chaos.chaos_pipeline, eop.chaos.chaos_pipeline, eop.chaos.z1_chaos_test]
        kwargs_list = [None, {'denoise': True}, None]
        return metrics_functions, metrics_name_list, kwargs_list
    
    print(f'Error in metric selection, name {name} is not a valid option')
    return None, None, None