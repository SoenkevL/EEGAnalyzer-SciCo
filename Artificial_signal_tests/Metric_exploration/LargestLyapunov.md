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
    *   Use `sklearn.metrics.pairwise.euclidean_distanes` to compute the pairwise Euclidean distances between the rows of the embedded matrix. The result is a square matrix (`dists`) where `dists[i, j]` is the Euclidean distance between the i-th and j-th points in the phase space.
 *  TODO: gemini skipped the details of how this is done. It uses a frequency decomposition to find an average frequency of the signal...
 

 ## 4. Exclude Temporal Neighbors
 

 *   **Purpose:** Avoid spurious correlations by excluding points that are too close in time when searching for nearest neighbors.
 *   **Steps:**
    *   For each point in the phase space (each row `i` in the `dists` matrix):
        *   Set the distances to points within a temporal window of `separation` points before and after the current point to infinity (`np.inf`). This prevents these points from being considered as nearest neighbors.
 

 ## 5. Find Nearest Neighbors
 

 *   **Purpose:** For each point in the trajectory, find its nearest neighbor in the phase space.
 *   **Steps:**
    *   Determine the number of trajectory points to consider (`ntraj`) based on the trajectory length (`len_trajectory`).
    *   Use `np.argmin` on the `dists` matrix (up to `ntraj` rows and columns) to find the index of the nearest neighbor for each point. The `axis=1` argument ensures that the minimum is found along each row (i.e., for each point).
    *   Store the indices of the nearest neighbors in the `min_dist_indices` array.
 

 ## 6. Track Trajectories and Calculate Divergence
 

 *   **Purpose:** Follow the trajectories of the original points and their nearest neighbors over time and calculate their average divergence.
 *   **Steps:**
    *   Initialize an array `trajectories` to store the average divergence at each time step.
    *   Iterate over the trajectory length (`len_trajectory`):
        *   For each time step `k`, calculate the distance between each point and its nearest neighbor at that time step. This is done by indexing the `dists` matrix using `(np.arange(ntraj) + k, min_dist_indices + k)`.
        *   Find the indices where the divergence is non-zero (`dist_nonzero`).
        *   If all divergences are zero, set the trajectory value to `-np.inf` to avoid errors.
        *   Otherwise, calculate the mean of the natural logarithm of the non-zero divergences and store it in the `trajectories` array.
 

 ## 7. Estimate Largest Lyapunov Exponent
 

 *   **Purpose:** Estimate the largest Lyapunov exponent from the average divergence of trajectories.
 *   **Steps:**
    *   Remove any infinite values from the `trajectories` array, storing the finite values in `divergence_rate`.
    *   Perform a linear least-squares fit to the `divergence_rate` data using `np.polyfit`. The x-values are the time steps (1 to the length of `divergence_rate`), and the y-values are the corresponding divergence rates.
    *   The slope of the fitted line is an estimate of the largest Lyapunov exponent.
 

 ## 8. Store Results
 

 *   **Purpose:** Store the calculated parameters and divergence rate for further analysis.
 *   **Steps:**
    *   Create a dictionary `parameters` to store the trajectory length and the divergence rate.
 

 This detailed breakdown should provide a clear understanding of the steps involved in Rosenstein's method for calculating the largest Lyapunov exponent.


# Some additional explanation
 # Deep Dive into Trajectory Tracking and Lyapunov Exponent Estimation in Rosenstein's Method
 

 This document provides a more detailed explanation of steps 5 through 8 in Rosenstein's method for calculating the largest Lyapunov exponent. These steps focus on finding nearest neighbors in the reconstructed phase space, tracking their trajectories, and estimating the Lyapunov exponent from their divergence.
 

 ## Recap: Steps 1-4
 

 Before diving into the details, let's briefly recap the initial steps:
 

 1.  **Input Validation:** Ensures the input signal is long enough for reliable calculations.
 2.  **Phase Space Reconstruction (Embedding):** Creates a higher-dimensional representation of the system's dynamics using time-delayed embedding.  This results in an `embedded` matrix where each row is a point in the reconstructed phase space.
 3.  **Pairwise Distance Matrix:** Calculates the Euclidean distances between all pairs of points in the embedded space, stored in the `dists` matrix. `dists[i, j]` represents the distance between point `i` and point `j`.
 4.  **Exclude Temporal Neighbors:** Sets distances to points that are too close in time to infinity to avoid spurious correlations.
 

 ## 5. Finding Nearest Neighbors: The Core Idea
 

 *   **Purpose:**  For each point in the *beginning* of the trajectory, find its closest neighbor in the reconstructed phase space.  This is the starting point for tracking how nearby trajectories diverge.
 *   **Why is this important?** The Lyapunov exponent measures the *average* rate of divergence of *nearby* trajectories. We need to identify these nearby trajectories to measure their divergence.
 *   **Steps:**
    *   **`ntraj = m - len_trajectory + 1`:**  This line calculates the number of trajectory starting points (`ntraj`) we'll consider.  `m` is the total number of embedded points. We only consider the first `ntraj` points because we need enough subsequent points to follow the trajectory for a length of `len_trajectory`.  If we started too late, we wouldn't have enough points to track the trajectory.
    *   **`min_dist_indices = np.argmin(dists[:ntraj, :ntraj], axis=1)`:** This is the key line. Let's break it down:
        *   `dists[:ntraj, :ntraj]` :  We're only considering the first `ntraj` rows and columns of the `dists` matrix.  This is because we only want to find neighbors for the *starting points* of our trajectories.  We're also restricting the *search space* for neighbors to these same starting points.  This is a common (though not universally required) practice in Rosenstein's method.  It helps to ensure that we're comparing trajectories that are evolving from similar regions of the phase space.
        *   `np.argmin(..., axis=1)`: For each row in the sliced `dists` matrix, `np.argmin` finds the index of the minimum value.  Since each row represents the distances from a particular point to all other points, `np.argmin` finds the index of the *nearest neighbor* for that point.  `axis=1` ensures that the minimum is found along each row.
        *   `min_dist_indices`:  This array stores the indices of the nearest neighbors for each of the `ntraj` starting points.  For example, if `min_dist_indices[0] == 5`, it means that the nearest neighbor to the first point in the embedded space (index 0) is the point at index 5.
    *   **`min_dist_indices = min_dist_indices.astype(int)`:**  Ensures that the indices are integers.
 

 ## 6. Tracking Trajectories and Calculating Divergence: Measuring Separation
 

 *   **Purpose:**  Follow the trajectories of each point and its nearest neighbor over time and calculate how their distance diverges.
 *   **Steps:**
    *   **`trajectories = np.zeros(len_trajectory)`:** Initializes an array to store the average divergence at each time step.
    *   **`for k in range(len_trajectory):`:**  This loop iterates over the trajectory length.  `k` represents the time step along the trajectory.
        *   **`divergence = dists[(np.arange(ntraj) + k, min_dist_indices + k)]`:** This is the most complex line in this section.  It calculates the distance between each point and its nearest neighbor *at time step k*.
            *   `np.arange(ntraj) + k`: Creates an array of indices representing the original points' positions *after k time steps*.  For example, if `ntraj = 5` and `k = 2`, this would be `[2, 3, 4, 5, 6]`.
            *   `min_dist_indices + k`: Creates an array of indices representing the nearest neighbors' positions *after k time steps*.  For example, if `min_dist_indices = [10, 12, 15, 18, 20]` and `k = 2`, this would be `[12, 14, 17, 20, 22]`.
            *   `dists[(..., ...)]`:  Uses these two arrays of indices to access the `dists` matrix.  It effectively retrieves the distances between the original points and their nearest neighbors *after k time steps*.
        *   **`dist_nonzero = np.where(divergence != 0)[0]`:** Finds the indices where the divergence is non-zero.  This is important because the logarithm of zero is undefined.
        *   **`if len(dist_nonzero) == 0:`:**  Handles the case where all divergences are zero (which can happen if the trajectories converge perfectly).  In this case, the trajectory value is set to `-np.inf` to avoid errors.
        *   **`else: trajectories[k] = np.mean(np.log(divergence[dist_nonzero]))`:**  Calculates the average divergence at time step `k`.  It takes the natural logarithm of the non-zero divergences and then calculates the mean.  The logarithm is used because the Lyapunov exponent measures the *exponential* rate of divergence.
 

 ## 7. Estimating the Largest Lyapunov Exponent: Finding the Slope
 

 *   **Purpose:** Estimate the largest Lyapunov exponent from the average divergence of trajectories.
 *   **Steps:**
    *   **`divergence_rate = trajectories[np.isfinite(trajectories)]`:** Removes any infinite values from the `trajectories` array.  These infinite values can occur if trajectories converge perfectly or if there are numerical issues.
    *   **`slope, intercept = np.polyfit(np.arange(1, len(divergence_rate) + 1), divergence_rate, 1)`:** Performs a linear least-squares fit to the `divergence_rate` data.
        *   `np.arange(1, len(divergence_rate) + 1)`: Creates an array of x-values representing the time steps (1, 2, 3, ...).
        *   `divergence_rate`: The y-values are the corresponding divergence rates.
        *   `np.polyfit(x, y, 1)`: Fits a line (degree 1 polynomial) to the data.  It returns the slope and intercept of the best-fit line.
    *   **`slope`:** The slope of the fitted line is an estimate of the largest Lyapunov exponent.  The slope represents the average rate of divergence of nearby trajectories.
 

 ## 8. Storing Results
 

 *   **Purpose:** Store the calculated parameters and divergence rate for further analysis.
 *   **Steps:**
    *   A dictionary `parameters` is created to store the trajectory length and the divergence rate.
 

 ## Key Concepts Revisited
 

 *   **Nearest Neighbors:** Finding points in the phase space that are initially close to each other.
 *   **Trajectory Tracking:** Following the evolution of these nearby points over time.
 *   **Divergence:** Measuring how the distance between these points changes over time.
 *   **Lyapunov Exponent:** Quantifying the average rate of exponential divergence, estimated from the slope of the divergence curve.
 

 By understanding these steps in detail, you can gain a deeper appreciation for how Rosenstein's method estimates the largest Lyapunov exponent, a crucial indicator of chaotic behavior in dynamical systems.

