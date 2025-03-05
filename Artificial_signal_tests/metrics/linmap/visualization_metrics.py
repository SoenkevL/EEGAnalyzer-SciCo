import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace `file_path` with the actual path to your file)
df = pd.read_csv("filtered_file.csv")

# Remove columns if their names contain 'unnamed'
df = df.loc[:, ~df.columns.str.contains('unnamed', case=False)]

# Extract the metrics and `r` columns
r_columns = [col for col in df.columns if col not in ['label', 'startDataRecord', 'duration', 'metric']]
data = df.set_index('metric')[r_columns]


# Function to handle missing data
def handle_missing_data(df, replacement_value=0, nan_threshold=0.3):
    """
    Handle missing data in the dataframe by dropping rows with NaN values
    exceeding a threshold or replacing them with a specified value.

    Args:
        df (pd.DataFrame): The input dataframe.
        replacement_value (int, float): Value to replace NaNs with.
        nan_threshold (float): Maximum proportion of NaNs allowed per row;
                               rows exceeding this will be removed.

    Returns:
        pd.DataFrame: Dataframe after handling missing values.
    """
    # Calculate percentage of NaNs per row
    nan_percentage = df.isnull().mean(axis=1)

    # Print rows with NaNs and their percentage
    rows_with_nans = df[nan_percentage > 0]
    print("Rows with NaN values and their percentage:")
    print(rows_with_nans.assign(nan_percentage=nan_percentage * 100))

    # Remove rows with NaN percentage above the threshold
    df_cleaned = df[nan_percentage <= nan_threshold]

    # Replace remaining NaNs with the replacement value
    df_cleaned = df_cleaned.fillna(replacement_value)

    return df_cleaned


# Handle missing data
imputed_data = handle_missing_data(data, replacement_value=0, nan_threshold=0.3)


# Line plot of each metric across `r`
plt.figure(figsize=(12, 6))
for index, row in imputed_data.iterrows():
    plt.plot(r_columns, row, label=index)
plt.xlabel("r (growth factor)")
plt.ylabel("Metric Value")
plt.title("Metrics Across r Values")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()

# Optional: Heatmap for easier visualization
plt.figure(figsize=(12, 6))
sns.heatmap(imputed_data, cmap="coolwarm", xticklabels=r_columns, yticklabels=imputed_data.index, cbar_kws={'label': 'Metric Value'})
plt.title("Heatmap of Metrics Across r")
plt.xlabel("r (growth factor)")
plt.ylabel("Metric")
plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(imputed_data)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust clusters based on your hypothesis
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the dataframe
imputed_data['cluster'] = clusters

# Visualize the clusters using PCA for dimension reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', label=clusters)
plt.colorbar(scatter, label="Cluster")
plt.title("Cluster Analysis of Metrics Across r")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

import numpy as np


# Generate bifurcation diagram
def logistic_map(r, x, n):
    for _ in range(n):
        x = r * x * (1 - x)
        yield x


r_values = np.linspace(0.5, 1.0, 500)
bifurcation_x = []
bifurcation_r = []
for r in r_values:
    x = np.random.random()
    orbit = list(logistic_map(r, x, 1000))[-100:]  # Last 100 values after transients
    bifurcation_x.extend(orbit)
    bifurcation_r.extend([r] * len(orbit))

# Plot bifurcation diagram
plt.figure(figsize=(12, 6))
plt.scatter(bifurcation_r, bifurcation_x, s=0.5, alpha=0.7, color='black')
plt.title("Bifurcation Diagram of the Logistic Map")
plt.xlabel("r (growth factor)")
plt.ylabel("x (state)")

# Overlay metrics for visual comparison
for metric, row in imputed_data.iterrows():
    plt.plot(r_columns, row, label=metric)
plt.legend(loc='best')
plt.show()

# Identify metrics with all constant values
constant_metrics = df.loc[(imputed_data.nunique(axis=1) <= 1)].index.tolist()
print("Metrics with constant values:", constant_metrics)

# Identify columns or rows with missing data
missing_values = imputed_data.isnull().sum()
print("Number of missing values in each r column:")
print(missing_values)



# Visualize missing values across r columns
plt.figure(figsize=(12, 6))
sns.heatmap(imputed_data.isnull(), cbar=False, cmap='viridis', xticklabels=r_columns, yticklabels=imputed_data.index)
plt.title("Missing Values in Metrics Across r")
plt.xlabel("Columns representing r (growth factor)")
plt.ylabel("Metrics (row index)")
plt.tight_layout()
plt.show()


# Extract a subset of r values
r_subset = [col for col in r_columns if 0.7 <= float(col) <= 0.9]
subset_data = imputed_data[r_subset]

# Plot metrics for this subset
plt.figure(figsize=(12, 6))
for index, row in subset_data.iterrows():
    plt.plot(r_subset, row, label=index)
plt.xlabel("r (growth factor)")
plt.ylabel("Metric Value")
plt.title("Subset of Metrics Across r (0.7 <= r <= 0.9)")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()


# Select Lyapunov exponent
lyapunov_data = df[df['metric'] == 'largest_lyapunov_exponent'].iloc[:, 4:].values[0]

# Plot the evolution of the largest Lyapunov exponent
plt.figure(figsize=(12, 6))
plt.plot(r_columns, lyapunov_data, marker='o', linestyle='-', label='Largest Lyapunov Exponent')
plt.axhline(y=0, color='r', linestyle='--', label='Chaotic Threshold')
plt.xlabel("r (growth factor)")
plt.ylabel("Largest Lyapunov Exponent")
plt.title("Largest Lyapunov Exponent Across r")
plt.legend()
plt.show()