from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise
import sklearn.neighbors
from icecream import ic
from neurokit2 import signal_psd, complexity_embedding, find_knee, NeuroKitWarning
from sklearn.decomposition import PCA
#%% complexity implementation


def complexity_lyapunov(
        signal,
        delay=1,
        dimension=2,
        method="rosenstein1993",
        separation="auto",
        **kwargs,
):
    """**(Largest) Lyapunov Exponent (LLE)**

    Lyapunov exponents (LE) describe the rate of exponential separation (convergence or divergence)
    of nearby trajectories of a dynamical system. It is a measure of sensitive dependence on
    initial conditions, i.e. how quickly two nearby states diverge. A system can have multiple LEs,
    equal to the number of the dimensionality of the phase space, and the largest LE value, "LLE" is
    often used to determine the overall predictability of the dynamical system.

    Different algorithms exist to estimate these indices:

    * **Rosenstein et al.'s (1993)** algorithm was designed for calculating LLEs from small
      datasets. The time series is first reconstructed using a delay-embedding method, and the
      closest neighbour of each vector is computed using the euclidean distance. These two
      neighbouring points are then tracked along their distance trajectories for a number of data
      points. The slope of the line using a least-squares fit of the mean log trajectory of the
      distances gives the final LLE.
    * **Makowski** is a custom modification of Rosenstein's algorithm, using KDTree for more
      efficient nearest neighbors computation. Additionally, the LLE is computed as the slope up to
      the changepoint of divergence rate (the point where it flattens out), making it more robust
      to the length trajectory parameter.
    * **Eckmann et al. (1986)** computes LEs by first reconstructing the time series using a
      delay-embedding method, and obtains the tangent that maps to the reconstructed dynamics using
      a least-squares fit, where the LEs are deduced from the tangent maps.

    .. warning::

      The **Eckman (1986)** method currently does not work. Please help us fixing it by double
      checking the code, the paper and helping us figuring out what's wrong. Overall, we would like
      to improve this function to return for instance all the exponents (Lyapunov spectrum),
      implement newer and faster methods (e.g., Balcerzak, 2018, 2020), etc. If you're interested
      in helping out with this, please get in touch!

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter. If method
        is ``"eckmann1986"``, larger values for dimension are recommended.
    method : str
        The method that defines the algorithm for computing LE. Can be one of ``"rosenstein1993"``,
        ``"makowski"``, or ``"eckmann1986"``.
    len_trajectory : int
        Applies when method is ``"rosenstein1993"``. The number of data points in which
        neighboring trajectories are followed.
    matrix_dim : int
        Applies when method is ``"eckmann1986"``. Corresponds to the number of LEs to return.
    min_neighbors : int, str
        Applies when method is ``"eckmann1986"``. Minimum number of neighbors. If ``"default"``,
        ``min(2 * matrix_dim, matrix_dim + 4)`` is used.
    **kwargs : optional
        Other arguments to be passed to ``signal_psd()`` for calculating the minimum temporal
        separation of two neighbors.

    Returns
    --------
    lle : float
        An estimate of the largest Lyapunov exponent (LLE) if method is ``"rosenstein1993"``, and
        an array of LEs if ``"eckmann1986"``.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute LLE.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=5, sampling_rate=100, frequency=[5, 8], noise=0.1)

      # Rosenstein's method
      @savefig p_complexity_lyapunov1.png scale=100%
      lle, info = nk.complexity_lyapunov(signal, method="rosenstein", show=True)
      @suppress
      plt.close()

      lle

      # Makowski's change-point method
      @savefig p_complexity_lyapunov2.png scale=100%
      lle, info = nk.complexity_lyapunov(signal, method="makowski", show=True)
      @suppress
      plt.close()

      # Eckman's method is broken. Please help us fix-it!
      # lle, info = nk.complexity_lyapunov(signal, dimension=2, method="eckmann1986")

    References
    ----------
    * Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical method
      for calculating largest Lyapunov exponents from small data sets.
      Physica D: Nonlinear Phenomena, 65(1-2), 117-134.
    * Eckmann, J. P., Kamphorst, S. O., Ruelle, D., & Ciliberto, S. (1986). Liapunov
      exponents from time series. Physical Review A, 34(6), 4971.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Compute Minimum temporal separation between two neighbors
    # -----------------------------------------------------------
    # Rosenstein (1993) finds a suitable value by calculating the mean period of the data,
    # obtained by the reciprocal of the mean frequency of the power spectrum.

    # "We impose the additional constraint that nearest neighbors have a temporal separation
    # greater than the mean period of the time series: This allows us to consider each pair of
    # neighbors as nearby initial conditions for different trajectories."

    # "We estimated the mean period as the reciprocal of the mean frequency of the power spectrum,
    # although we expect any comparable estimate, e.g., using the median frequency of the magnitude
    # spectrum, to yield equivalent results."
    if separation == "auto":
        # Actual sampling rate does not matter
        psd = signal_psd(
            signal, sampling_rate=1000, method="fft", normalize=False, show=False
        )
        mean_freq = np.sum(psd["Power"] * psd["Frequency"]) / np.sum(psd["Power"])

        # 1 / mean_freq = seconds per cycle
        separation = int(np.ceil(1 / mean_freq * 1000))
    else:
        assert isinstance(separation, int), "'separation' should be an integer."

    # Run algorithm
    # ----------------
    # Method
    method = method.lower()
    if method in ["rosenstein", "rosenstein1993"]:
        le, parameters = _complexity_lyapunov_rosenstein(
            signal, delay, dimension, separation, **kwargs
        )
    elif method in ["makowski"]:
        le, parameters = _complexity_lyapunov_makowski(
            signal, delay, dimension, separation, **kwargs
        )
    # elif method in ["eckmann", "eckmann1986", "eckmann1986"]:
    #     le, parameters = _complexity_lyapunov_eckmann(
    #         signal,
    #         dimension=dimension,
    #         separation=separation,
    #     )
    else:
        raise ValueError(
            "NeuroKit error: complexity_lyapunov(): 'method' should be one of "
            " 'rosenstein1993', 'makowski', 'eckmann1986'."
        )

    # Store params
    info = {
        "Dimension": dimension,
        "Delay": delay,
        "Separation": separation,
        "Method": method,
    }
    info.update(parameters)

    return le, info


# =============================================================================
# Methods
# =============================================================================


def _complexity_lyapunov_makowski(
        signal,
        delay=1,
        dimension=2,
        separation=1,
        max_length="auto",
        show=False,
):
    # Store parameters
    info = {
        "Dimension": dimension,
        "Delay": delay,
    }

    # Embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    n = len(embedded)

    # Set the maxiimum trajectory length to 10 times the delay
    if max_length == "auto":
        max_length = int(delay * 10)
    if max_length >= n / 2:
        max_length = n // 2

    # Create KDTree and query for nearest neighbors
    tree = sklearn.neighbors.KDTree(embedded, metric="euclidean")

    # Query for nearest neighbors. To ensure we get a neighbor outside of the `separation`,
    # k=1 is the point itself, k=2 is the nearest neighbor, and k=3 is the second nearest neighbor.
    idx = tree.query(embedded, k=2 + separation, return_distance=False)

    # The neighbor outside the `separation` region will be the last one in the returned list.
    idx = idx[:, -1]

    # Compute the average divergence for each trajectory length
    trajectories = np.zeros(max_length)
    for k in range(1, max_length + 1):
        valid = np.where((np.arange(n - k) + k < n) & (idx[: n - k] + k < n))[0]

        if valid.size == 0:
            trajectories[k - 1] = -np.inf
            continue

        divergences = np.linalg.norm(
            embedded[valid + k] - embedded[idx[valid] + k],
            axis=1,
            )
        divergences = divergences[divergences > 0]
        if len(divergences) == 0:
            trajectories[k - 1] = np.nan
        else:
            trajectories[k - 1] = np.mean(np.log(divergences))

    # Change point
    x_axis = range(1, len(trajectories) + 1)
    knee = find_knee(y=trajectories, x=x_axis, show=False, verbose=False)
    info["Divergence_Rate"] = trajectories
    info["Changepoint"] = knee

    # Linear fit
    slope, intercept = np.polyfit(x_axis[0:knee], trajectories[0:knee], 1)
    if show is True:
        plt.plot(np.arange(1, len(trajectories) + 1), trajectories)
        plt.axvline(knee, color="red", label="Changepoint", linestyle="--")
        plt.axline(
            (0, intercept), slope=slope, color="orange", label="Least-squares Fit"
        )
        plt.ylim(bottom=np.min(trajectories))
        plt.ylabel("Divergence Rate")
        plt.title(f"Largest Lyapunov Exponent (slope of the line) = {slope:.3f}")
        plt.legend()
    return slope, info


def _complexity_lyapunov_rosenstein(
        signal, delay=1, dimension=2, separation=1, len_trajectory=20, show=False, **kwargs
):
    # 1. Check that sufficient data points are available
    print(f'step: # 1. Check that sufficient data points are available')
    # Minimum length required to find single orbit vector
    min_len = (dimension - 1) * delay + 1
    # We need len_trajectory orbit vectors to follow a complete trajectory
    min_len += len_trajectory - 1
    # we need tolerance * 2 + 1 orbit vectors to find neighbors for each
    min_len += separation * 2 + 1
    # Sanity check
    if len(signal) < min_len:
        warn(
            f"for dimension={dimension}, delay={delay}, separation={separation} and "
            f"len_trajectory={len_trajectory}, you need at least {min_len} datapoints in your"
            " time series.",
            category=NeuroKitWarning,
        )
    print(
        f"for dimension={dimension}, delay={delay}, separation={separation} and "
        f"len_trajectory={len_trajectory}, you need at least {min_len} datapoints in your"
        f" time series, you have {len(signal)}",)

    # Embedding
    # opt_delay = nk.complexity_delay(signal)
    # opt_dim = nk.complexity_dimension(signal)
    # opt_embedded = complexity_embedding(signal, delay=opt_delay[0], dimension=opt_dim[0])
    # pca = pca(n_components=3)
    # principal_components = pca.fit_transform(opt_embedded)
    # embedded = principal_components
    # 2. Create the embedding for the signal
    print(f'step: # 2. Create the embedding for the signal')
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension, show=False)
    m = len(embedded)
    print(f'Created embedded matrix of dimension=({np.shape(embedded)})')

    # Construct matrix with pairwise distances between vectors in orbit
    print(f'step: # 3. Create a pointwise distance matrix of the reconstructed phase space')
    dists = sklearn.metrics.pairwise.euclidean_distances(embedded)

    print(f'step: # 4. Ensure sufficient temporal seperation of neighbors')
    for i in range(m):
        # Exclude indices within separation
        dists[i, max(0, i - separation) : i + separation + 1] = np.inf

    # Find indices of nearest neighbours
    print(f'step: # 5. Find the  nearest spatial neighbor for every point of the phase space')
    ntraj = m - len_trajectory + 1
    min_dist_indices = np.argmin(
        dists[:ntraj, :ntraj], axis=1
    )  # exclude last few indices
    min_dist_indices = min_dist_indices.astype(int)

    # Follow trajectories of neighbour pairs for len_trajectory data points
    print(f'step # 6. For a trajectory_length={len_trajectory} calculate the divergence between all spatial nearest neighbor points')
    trajectories = np.zeros(len_trajectory)
    for k in range(len_trajectory):
        divergence = dists[(np.arange(ntraj) + k, min_dist_indices + k)]
        dist_nonzero = np.where(divergence != 0)[0]
        if len(dist_nonzero) == 0:
            trajectories[k] = -np.inf
        else:
            # Get average distances of neighbour pairs along the trajectory
            trajectories[k] = np.mean(np.log(divergence[dist_nonzero]))

    divergence_rate = trajectories[np.isfinite(trajectories)]

    # LLE obtained by least-squares fit to average line
    slope, intercept = np.polyfit(
        np.arange(1, len(divergence_rate) + 1), divergence_rate, 1
    )

    # Store info
    parameters = {
        "Trajectory_Length": len_trajectory,
        "Divergence_Rate": divergence_rate,
    }

    if show is True:
        plt.plot(np.arange(1, len(divergence_rate) + 1), divergence_rate)
        plt.axline(
            (0, intercept), slope=slope, color="orange", label="Least-squares Fit"
        )
        plt.ylabel("Divergence Rate")
        plt.title(f"Largest Lyapunov Exponent (slope of the line) = {slope:.3f}")
        plt.legend()

    return slope, parameters


#%% extracting the traces

import neurokit2 as nk

signal = nk.signal_simulate(duration=5, sampling_rate=200, frequency=[5, 6], noise=0.3, random_state=42)
delay = 1
dimension = 3

# Rosenstein's method
embedded = complexity_embedding(signal, delay=delay, dimension=dimension, show=True)
plt.title(f'Embedding of the signal that is processed\ndelay={delay} and dimension={dimension}')
plt.show()
lle, info = complexity_lyapunov(signal, method="rosenstein", dimension=dimension, delay=delay,  show=True)
plt.show()

#%%
opt_delay = nk.complexity_delay(signal, show=True)
plt.show()
opt_dim = nk.complexity_dimension(signal, delay=opt_delay[0], show=True)
plt.show()
opt_dim2 = nk.complexity_dimension(signal, delay=1, show=True)
plt.show()
opt_embedded = complexity_embedding(signal, delay=opt_delay[0], dimension=opt_dim[0], show=True)
plt.title(f'Embedding with joint optimal parameter estimation\n'
          f'delay={opt_delay[0]} and dimension={opt_dim[0]}\n'
          f'shown are just the last three dimensions')
plt.tight_layout()
plt.show()
opt_embedded = complexity_embedding(signal, delay=opt_delay[0], dimension=opt_dim2[0], show=True)
plt.title(f'Embedding with independent optimal parameter estimation\n'
          f'delay={opt_delay[0]} and dimension={opt_dim2[0]}\n'
          f'shown are just the last three dimensions')
plt.tight_layout()
plt.show()


#instead of plotting the first three dimensions we will create the pca of the components and plot those
pca = PCA(n_components=3)
principal_components = pca.fit_transform(opt_embedded)
