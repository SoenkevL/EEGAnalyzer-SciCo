#%% defintion from neurokit2
import numpy as np
import pandas as pd

from neurokit2 import complexity_ordinalpatterns, complexity_symbolize


def complexity_lempelziv(
        signal,
        delay=1,
        dimension=2,
        permutation=False,
        symbolize="mean",
        **kwargs,
):
    """**Lempel-Ziv Complexity (LZC, PLZC and MSLZC)**

    Computes Lempel-Ziv Complexity (LZC) to quantify the regularity of the signal, by scanning
    symbolic sequences for new patterns, increasing the complexity count every time a new sequence
    is detected. Regular signals have a lower number of distinct patterns and thus have low LZC
    whereas irregular signals are characterized by a high LZC. While often being interpreted as a
    complexity measure, LZC was originally proposed to reflect randomness (Lempel and Ziv, 1976).

    Permutation Lempel-Ziv Complexity (**PLZC**) combines LZC with :func:`permutation <entropy_permutation>`.
    A sequence of symbols is generated from the permutations observed in the :func:`tine-delay
    embedding <complexity_embedding>`, and LZC is computed over it.

    Multiscale (Permutation) Lempel-Ziv Complexity (**MSLZC** or **MSPLZC**) combines permutation
    LZC with the :func:`multiscale approach <entropy_multiscale>`. It first performs a
    :func:`coarse-graining <complexity_coarsegraining>` procedure to the original time series.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter. Only used
        when ``permutation=True``.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter. Only used
        when ``permutation=True``
    permutation : bool
        If ``True``, will return PLZC.
    symbolize : str
        Only used when ``permutation=False``. Method to convert a continuous signal input into a
        symbolic (discrete) signal. By default, assigns 0 and 1 to values below and above the mean.
        Can be ``None`` to skip the process (in case the input is already discrete). See
        :func:`complexity_symbolize` for details.
    **kwargs
        Other arguments to be passed to :func:`complexity_ordinalpatterns` (if
        ``permutation=True``) or :func:`complexity_symbolize`.

    Returns
    ----------
    lzc : float
        Lempel Ziv Complexity (LZC) of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute LZC.

    See Also
    --------
    .complexity_symbolize, .complexity_ordinalpatterns, .entropy_permutation,

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # LZC
      lzc, info = nk.complexity_lempelziv(signal)
      lzc

      # PLZC
      plzc, info = nk.complexity_lempelziv(signal, delay=1, dimension=3, permutation=True)
      plzc

    .. ipython:: python

      # MSLZC
      @savefig p_complexity_lempelziv1.png scale=100%
      mslzc, info = nk.entropy_multiscale(signal, method="LZC", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # MSPLZC
      @savefig p_complexity_lempelziv2.png scale=100%
      msplzc, info = nk.entropy_multiscale(signal, method="LZC", permutation=True, show=True)
      @suppress
      plt.close()

    References
    ----------
    * Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences. IEEE Transactions on
      information theory, 22(1), 75-81.
    * Nagarajan, R. (2002). Quantifying physiological data with Lempel-Ziv complexity-certain
      issues. IEEE Transactions on Biomedical Engineering, 49(11), 1371-1373.
    * Kaspar, F., & Schuster, H. G. (1987). Easily calculable measure for the complexity of
      spatiotemporal patterns. Physical Review A, 36(2), 842.
    * Zhang, Y., Hao, J., Zhou, C., & Chang, K. (2009). Normalized Lempel-Ziv complexity and
      its application in bio-sequence analysis. Journal of mathematical chemistry, 46(4), 1203-1212.
    * Bai, Y., Liang, Z., & Li, X. (2015). A permutation Lempel-Ziv complexity measure for EEG
      analysis. Biomedical Signal Processing and Control, 19, 102-114.
    * Borowska, M. (2021). Multiscale Permutation Lempel-Ziv Complexity Measure for Biomedical
      Signal Analysis: Interpretation and Application to Focal EEG Signals. Entropy, 23(7), 832.

    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Store parameters
    info = {"Permutation": permutation}

    # Permutation or not
    if permutation:
        info["Dimension"] = dimension
        info["Delay"] = delay
        # Permutation on the signal (i.e., converting to ordinal pattern).
        _, info = complexity_ordinalpatterns(signal, delay=delay, dimension=dimension, **kwargs)
        symbolic = info["Uniques"]
    else:
        # Binarize the signal
        symbolic = complexity_symbolize(signal, method=symbolize, **kwargs)
        info['symbolic'] = symbolic

    # Count using the lempelziv algorithm
    info["Complexity_Kolmogorov"], n, info['Tokens'], info['state_trace'] = _complexity_lempelziv_count(symbolic)

    # Normalize
    if permutation is False:
        lzc = (info["Complexity_Kolmogorov"] * np.log2(n)) / n
    else:
        lzc = (
                      info["Complexity_Kolmogorov"] * np.log2(n) / np.log2(np.math.factorial(dimension))
              ) / n

    return lzc, info


# =============================================================================
# Utilities
# =============================================================================
def _complexity_lempelziv_count(symbolic):
    """Computes LZC counts from symbolic sequences"""

    # TODO: I really can't imagine that there is no faster way of doing that that with a while loop

    # Convert to string (faster)
    string = "".join(list(symbolic.astype(int).astype(str)))

    # Initialize variables
    n = len(string)
    tokens = []
    state_trace = []

    s = "0" + string #the symbolized string that we operate on
    complexity = 1 #the complexity of the series
    prefix_length = 1
    pointer = 0 #counter to traverse along the SB (search buffer)
    length_component = 1 #length of the current pattern that has been matched in the LA part
    max_length_component = 1 #length of the longest pattern that has been matched in the LA (look ahead) part
    stop = False
    comparison = [(pointer + length_component, s[pointer + length_component]),
                  (prefix_length + length_component, s[prefix_length + length_component])]
    state_trace.append(param_to_dict(comparison,'Starting the process with the initial values',
                                     pointer, length_component, max_length_component, prefix_length, complexity, True))

    # Start counting
    while stop is False:
        comparison = [(pointer+length_component, s[pointer+length_component]), (prefix_length+length_component, s[prefix_length+length_component])]
        if s[pointer + length_component] != s[prefix_length + length_component]:
            state_trace.append(
                param_to_dict(comparison,
                              f'The compared bits did not match',
                              pointer, length_component, max_length_component, prefix_length, complexity, True)
            )
            if length_component > max_length_component:
                # k_max stores the length of the longest pattern in the LA that has been matched
                # somewhere in the SB
                max_length_old = max_length_component
                max_length_component = length_component
                state_trace.append(
                    param_to_dict(comparison, f'-> the current_component_length ({length_component}) is bigger'
                                              f'\n   than the current max_component_length ({max_length_old}).'
                                              f'\n-> increasing max_component_length to {max_length_component}.',
                                  pointer, length_component, max_length_component, prefix_length, complexity, False)
                )

            # we increase i while the bit doesn't match, looking for a previous occurrence of a
            # pattern. s[i+k] is scanning the "search buffer" (SB)
            pointer = pointer + 1
            state_trace.append(
                param_to_dict(comparison, '-> increasing the pointer to continue searching the search buffer (SB)'
                              , pointer, length_component, max_length_component, prefix_length, complexity, False)
            )
            # we stop looking when i catches up with the first bit of the "look-ahead" (LA) part.
            if pointer == prefix_length:
                # If we were actually compressing, we would add the new token here. here we just
                # count reconstruction STEPs
                complexity = complexity + 1
                # we add the prefix to the found prefixes
                tokens.append(s[prefix_length:prefix_length+max_length_component])
                prefix_length = prefix_length + max_length_component
                state_trace.append(
                    param_to_dict(comparison, '=> pointer reached look ahead (LA) part:\n'
                                              '   increasing complexity and adding current max_component_length to prefix length',
                                  pointer, length_component, max_length_component, prefix_length, complexity, False)
                )
                # if the LA surpasses length of string, then we stop.
                if prefix_length + 1 > n:
                    stop = True
                # after STEP,
                else:
                    # we reset the searching index to beginning of SB (beginning of string)
                    pointer = 0
                    # we reset pattern matching index. Note that we are actually matching against
                    # the first bit of the string, because we added an extra 0 above, so i+k is the
                    # first bit of the string.
                    length_component = 1
                    # and we reset max length of matched pattern to k.
                    max_length_component = 1
                    state_trace.append(
                        param_to_dict(comparison, '-> step finished before signal ends: \nResetting pointer to beginning of SB,'
                                                  '\n   reset component_length,'
                                                  '\n   and the max_component_length for next token search'
                                      , pointer, length_component, max_length_component, prefix_length, complexity, False)
                    )
            else:
                length_component = 1
                state_trace.append(
                    param_to_dict(comparison, '-> pointer did not catch up to LA: finished matching pattern in SB,'
                                              '\n reset component_length and continue',
                                  pointer, length_component, max_length_component, prefix_length, complexity, False)
                )
        # I increase k as long as the pattern matches, i.e. as long as s[j+k] bit string can be
        # reconstructed by s[i+k] bit string. Note that the matched pattern can "run over" j
        # because the pattern starts copying itself (see LZ 76 paper). This is just what happens
        # when you apply the cloning tool on photoshop to a region where you've already cloned...
        else:
            state_trace.append(
                param_to_dict(comparison, 'The bits match, increasing the matched pattern length',
                              pointer, length_component, max_length_component, prefix_length, complexity, True)
            )
            length_component = length_component + 1
            # if we reach the end of the string while matching, we need to add that to the tokens,
            # and stop.
            if prefix_length + length_component > n:
                complexity = complexity + 1
                tokens.append(s[prefix_length:prefix_length+prefix_length])
                state_trace.append(
                    param_to_dict(comparison, 'Reached the end of the sequence on a matching bit,\n'
                                              ' increasing complexity by 1',
                                  pointer, length_component, max_length_component, prefix_length, complexity, False)
                )
                stop = True
    print_states(symbolic, state_trace, tokens)
    print(tokens)
    return complexity, n, tokens, state_trace

# Custom functions to format the information that I extract from the lempel ziv complexity process
def param_to_dict(comparison, description, pointer, length_component, max_length_component, prefix_length, complexity, new_iter):
    return {
        'comparison': comparison,
        'description': description,
        'pointer': pointer,
        'length_component': length_component,
        'max_length_component': max_length_component,
        'prefix_length': prefix_length,
        'complexity': complexity,
        'new_iter': new_iter,
    }

def print_states(symbolic, state_trace, tokens) -> str:
    '''
    Function which produces a nicely formatted string for printing
    :param symbolic: the signal
    :param state_trace: the dictionary containing the state trace showing how the parameters change
    :param tokens: the list of tokens that were extracted from the signal
    :return: string
    '''
    stdf = pd.DataFrame(state_trace)
    stdf['complexity_diff'] = stdf.complexity.diff()
    stdf['max_length_component_diff'] = stdf.max_length_component.diff()
    stdf['length_component_diff'] = stdf.length_component.diff()
    stdf['pointer_diff'] = stdf.pointer.diff()
    stdf['prefix_length_diff'] = stdf.prefix_length.diff()
    max_description_length = stdf.description.max()
    strings = list(stdf.apply(print_row, signal=symbolic, axis=1))
    print(''.join(strings))
    stdf_short = stdf[['comparison', 'description', 'length_component', 'complexity', 'max_length_component']]
    return stdf, stdf_short, strings

def print_row(row, signal):
    signal = ''.join([str(x) for x in signal])
    pointer_str = '-' * max(0, row.pointer) + '^' * row.length_component
    prefix_str = '-' * max(0, row.prefix_length) + 'v' * row.length_component
    string = ''
    if(row.new_iter):
        temp = ('\n'
                f'\n{prefix_str}'
                f'\n{signal}'
                f'\n{pointer_str}'
                f'\n{row.comparison}')
        string = string + temp
    string = string + '\n' +str(row.description)
    return string

#%% making a proper plot for the steps playing a bit with matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc_context
import numpy as np

def clean_plot(func):
    """Decorator to apply a temporary rc configuration for a clean plot."""
    def wrapper(*args, **kwargs):
        with rc_context(rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.left": False,
            "xtick.bottom": False,
            "ytick.left": False,
            "xtick.labelbottom": False,
            "ytick.labelleft": False
        }):
            result = func(*args, **kwargs)  # Call original function
            plt.tight_layout()
            return result
    return wrapper

@clean_plot
def plot_text(input_text_arr):
    input_text_arr_raw = [x if x else None for x in input_text_arr.split('\n')]
    input_text_arr = np.array(list(filter(None, input_text_arr_raw)))
    num_cols = input_text_arr.shape[0]
    """Plots each character in a string as separate text elements."""
    fig, ax = plt.subplots(figsize=(5, np.floor(num_cols/3)))
    for j, string in enumerate(input_text_arr):
        for i, char in enumerate(string):
            ax.text(i + 0.5, num_cols - j - 0.5, char, va='top', ha='center')
    max_text_len = max(len(string) for string in input_text_arr)
    ax.set_ylim(0, num_cols)
    ax.set_xlim(0, max_text_len)

def plot_text_2(input_text_arr):
    # input_text_arr = np.array([
    #     '[(0,0), (0,0)]',
    #     '-----^^^^',
    #     '000000111000000',
    #     '-----------v',
    #     'some random text'
    # ])
    num_cols = input_text_arr.shape[0]
    num_cols_freq = 0.9/(num_cols)
    max_description_len = max(len(string) for string in input_text_arr[3:])
    description_char_freq = 0.9/max_description_len
    signal_len = max(len(string) for string in input_text_arr[:3])
    signal_char_freq = 0.9/signal_len
    """Plots each character in a string as separate text elements."""
    fig, ax = plt.subplots(figsize=(5, num_cols))
    for j, string in enumerate(input_text_arr):
        if j:
            for i, char in enumerate(string):
                if j < 4:
                    ax.text(i*signal_char_freq + 0.05, j*num_cols_freq + 0.05, char, transform=ax.transAxes, va='center', ha='center')
                else:
                    ax.text(i*description_char_freq + 0.05, j*num_cols_freq + 0.05, char, transform=ax.transAxes, va='center', ha='center')
        else:
            ax.legend(string)
    return ax


#%% chat gpts definition of lempel ziv complexity (actually how I also understood it before studying code implementationsJ)
def lempel_ziv_complexity(signal):
    prefix_set = set()
    current_phrase = ""
    complexity = 0
    steps = []

    for i, bit in enumerate(signal):
        bit = str(bit)
        current_phrase += bit  # Extend the current phrase

        if current_phrase not in prefix_set:
            # New phrase detected
            prefix_set.add(current_phrase)
            complexity += 1
            steps.append(f"New phrase found: '{current_phrase}', increasing complexity to {complexity}")
            current_phrase = ""  # Reset for the next phrase

    return complexity, steps


#%% creating a signal and analyzing it
import neurokit2 as nk
import matplotlib.pyplot as plt

signal = nk.signal_simulate(duration=0.25, sampling_rate=200, frequency=[5, 6], noise=0.5, random_state=84)
plt.plot(signal)
# plt.show()

# LZC
lzc, info = complexity_lempelziv(signal[:10])

complexity, process_steps = lempel_ziv_complexity(signal[:10])

#%% creating the state iterator

#%% alternative version of lempel ziv complexity
print('\n\n\n'
      'This is a new string'
      '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
      '\n')
for step in process_steps:
    print(step)

print(f"Final Lempel-Ziv Complexity: {complexity}")
