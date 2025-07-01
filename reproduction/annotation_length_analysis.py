#%% imports
import mne
import pandas as pd
import os

#%% load data
folder = '/home/soenkevl/Documents/EEGAnalyzer/reproduction/EIRatio_BIDS'

#%% go through folder and extract the annotations, subject and condition into dataframes
annotation_dataframes = []
for root, dirs, files in os.walk(folder):
    for file in files:
        filepath = os.path.join(root, file)
        print(filepath)
        # identify relevant files
        if not 'exclude' in filepath and 'preprocessed-raw.fif' in filepath:
            raw = mne.io.read_raw_fif(filepath)
            info = raw.info
            annotations_df = raw.annotations.to_data_frame()
            info_from_file = file.split('_')
            annotations_df['subject'] = info_from_file[0].split('-')[1]
            annotations_df['condition'] = info_from_file[1].split('-')[1]
            annotations_df = annotations_df[['subject', 'condition', 'description', 'duration', 'onset']]
            annotation_dataframes.append(annotations_df)

#%% concatenate the individual dataframes into one
annotation_df = pd.concat(annotation_dataframes)
annotation_df = annotation_df.sort_values('subject').reset_index(drop=True)

#%% unique descriptions within the dataframe
unique_descriptions = annotation_df['description'].unique()

#%% reduce df to only contain the relevant descriptions
relevant_annot_df = annotation_df[annotation_df['description'].isin(['AWAKE-EC', 'ANES'])]
relevant_annot_df = relevant_annot_df.drop(columns=['onset'])

#%% group by subject and description and aggregate the duration and count of descriptions
gdf = relevant_annot_df.groupby(['subject', 'description']).agg({'duration': ['sum', 'count']}).reset_index()
annotation_descriptive_statistics = gdf.describe()
annotation_descriptive_statistics_per_condition = gdf.groupby('description').describe()

#%% check which subjects have less than 60 sec of annotations available
subects_with_less_than_60_sec = gdf[gdf['duration']['sum'] < 60]
