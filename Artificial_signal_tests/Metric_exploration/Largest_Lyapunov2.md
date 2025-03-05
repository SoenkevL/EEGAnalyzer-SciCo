# Rosenstein's Method for Largest Lyapunov Exponent Calculation

This document explains Rosenstein's method for calculating the largest Lyapunov exponent (LLE), a measure of the rate at which nearby trajectories diverge in a dynamical system. A positive LLE is a hallmark of chaotic behavior. The method involves reconstructing the system's dynamics in a phase space, identifying nearest neighbors, tracking their trajectories, and estimating the LLE from the average rate of divergence. The following sections provide a detailed breakdown of each step, with a particular focus on the process of finding and tracking nearest neighbors.

## 1. Input Validation and Data Sufficiency Check

*   **Purpose:** Ensure that the input signal has enough data points to perform the calculations reliably, given the chosen parameters.
*   **Steps:**
    *   Calculate the minimum required data length (`min_len`) based on the embedding dimension (`dimension`), time delay (`delay`), separation interval (`separation`), and trajectory length (`len_trajectory`). The formula accounts for:
        *   The number of points needed to form a single embedded vector: `(dimension - 1) * delay + 1`
        *   The number of embedded vectors needed to follow a trajectory: `len_trajectory - 1`
        *   The number of embedded vectors to exclude when searching for neighbors: `separation * 2 + 1`
    *   Compare `min_len` with the actual length of the input signal (`len(signal)`).
    *   If `len(signal) < min_len`, issue a warning indicating that the time series is too short for the specified parameters.

## 2. Phase Space Reconstruction (Embedding)

*   **Purpose:** Reconstruct the system's dynamics in a higher-dimensional space using the method of time delays.
*   **Steps:**
    *   Apply the `complexity_embedding` function to create an embedded matrix. Each row of the embedded matrix represents a point in the reconstructed phase space.
    *   The `delay` parameter determines the time lag between the components of each embedded vector.
    *   The `dimension` parameter determines the number of components in each embedded vector (i.e., the dimensionality of the reconstructed phase space).

## 3. Construct Pairwise Distance Matrix

*   **Purpose:** Calculate the Euclidean distances between all pairs of points in the reconstructed phase space.
*   **Steps:**
    *   Use `sklearn.metrics.pairwise.euclidean_distances` to compute the pairwise Euclidean distances between the rows of the embedded matrix. The result is a square matrix (`dists`) where `dists[i, j]` is the Euclidean distance between the i-th and j-th points in the phase space.

## 4. Exclude Temporal Neighbors

*   **Purpose:** Avoid spurious correlations by excluding points that are too close in time when searching for nearest neighbors.
*   **Steps:**
    *   For each point in the phase space (each row `i` in the `dists` matrix):
        *   Set the distances to points within a temporal window of `separation` points before and after the current point to infinity (`np.inf`). This prevents these points from being considered as nearest neighbors.

## 5. Find Nearest Neighbors

*   **Purpose:** For each point in the trajectory, find its nearest neighbor in the phase space.
*   **Steps:**
    *   Determine the number of trajectory points to consider (`ntraj`) based on the trajectory length (`len_trajectory`).  `ntraj = m - len_trajectory + 1` where `m` is the total number of embedded points. We only consider the first `ntraj` points because we need enough subsequent points to follow the trajectory for a length of `len_trajectory`.
    *   Use `np.argmin` on the `dists` matrix (up to `ntraj` rows and columns) to find the index of the nearest neighbor for each point. The `axis=1` argument ensures that the minimum is found along each row (i.e., for each point). Specifically, use `min_dist_indices = np.argmin(dists[:ntraj, :ntraj], axis=1)`. We're only considering the first `ntraj` rows and columns of the `dists` matrix because we only want to find neighbors for the *starting points* of our trajectories.  We're also restricting the *search space* for neighbors to these same starting points.
    *   Store the indices of the nearest neighbors in the `min_dist_indices` array.
    *   Ensure that the indices are integers: `min_dist_indices = min_dist_indices.astype(int)`.

## 6. Track Trajectories and Calculate Divergence

*   **Purpose:** Follow the trajectories of the original points and their nearest neighbors over time and calculate their average divergence.
*   **Steps:**
    *   Initialize an array `trajectories` to store the average divergence at each time step.
    *   Iterate over the trajectory length (`len_trajectory`):
        *   For each time step `k`, calculate the distance between each point and its nearest neighbor at that time step. This is done by indexing the `dists` matrix using `divergence = dists[(np.arange(ntraj) + k, min_dist_indices + k)]`.
        *   Find the indices where the divergence is non-zero (`dist_nonzero`).
        *   If all divergences are zero, set the trajectory value to `-np.inf` to avoid errors.
        *   Otherwise, calculate the mean of the natural logarithm of the non-zero divergences and store it in the `trajectories` array: `trajectories[k] = np.mean(np.log(divergence[dist_nonzero]))`. The logarithm is used because the Lyapunov exponent measures the *exponential* rate of divergence.
    *   The `trajectories` array now contains the average divergence at each time step.

## 7. Estimate Largest Lyapunov Exponent

*   **Purpose:** Estimate the largest Lyapunov exponent from the average divergence of trajectories.
*   **Steps:**
    *   Remove any infinite values from the `trajectories` array, storing the finite values in `divergence_rate`. `divergence_rate = trajectories[np.isfinite(trajectories)]`.
    *   Perform a linear least-squares fit to the `divergence_rate` data using `np.polyfit`. The x-values are the time steps (1 to the length of `divergence_rate`), and the y-values are the corresponding divergence rates. `slope, intercept = np.polyfit(np.arange(1, len(divergence_rate) + 1), divergence_rate, 1)`.
    *   The slope of the fitted line is an estimate of the largest Lyapunov exponent.

## 8. Store Results

*   **Purpose:** Store the calculated parameters and divergence rate for further analysis.
*   **Steps:**
    *   Create a dictionary `parameters` to store the trajectory length and the divergence rate.

