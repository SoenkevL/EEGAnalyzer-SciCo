import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'Logistic_metrics.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Extract all rows with the first 50 columns, keeping index and headers
df_first_50_columns = df.iloc[:, :50]

# Display the resulting DataFrame
print(df_first_50_columns)

# Optionally, save this filtered DataFrame back to a new CSV file
df_first_50_columns.to_csv('filtered_file.csv', index=True)