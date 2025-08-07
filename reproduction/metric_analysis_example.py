'''
This file serves as an example how to analyse the metrics that the pipeline produces
By running the pipeline beforehand we analysed the file PN001-preprocessed-raw.edf.
We used two different runs:
 1. common averaging, 0.5-12 Hz bandpass and 100 Hz sampling frequency.
 2. doublebanana, 0.5-60 Hz bandpass and 200 Hz sampling frequency.
In this script we show how:
 -we can load the created data for the two runs
 -show how the metric changed over time
 -compare metrics across runs to see differences due to the processing parameters
'''

#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mne
import multiprocessing as mp

#%% load the data from the two different runs we di
print(f'current working directory: {os.getcwd()}')
run1 = pd.read_csv('example/metrics/allFreq/doublebanana/PN001-preprocessed-metrics.csv')
run2 = pd.read_csv('example/metrics/lowfreq/avg/PN001-preprocessed-metrics.csv')
raw = mne.io.read_raw('example/eeg/PN001-preprocessed-raw.fif', preload=True)

#%% plot the loaded eeg file as a seperate process so It can be kept open while continuing the analysis of the metrics
def eeg_plot(raw):
    raw.plot(duration=20, remove_dc=False, block=True, show_options=True, title='Preprocessed EEG')
def mp_eeg_plot(raw):
    plot = (mp.Process(target=eeg_plot, args=(raw,)))
    plot.start()
    return plot
current_eeg_plot = mp_eeg_plot(raw)

#%% display the different metrics in a plot over time
all_metrics = ['fractal_dimension_katz',
                'fractal_dimension_higuchi_k-10',
                'fractal_dimension_hurst',
                'permutation_entropy',
                'multiscale_entropy',
                'multiscale_permutation_entropy',
                'lempel_ziv_complexity',
                'largest_lyapunov_exponent',
                 ]

plot_metrics = [
               'fractal_dimension_higuchi_k-10',
               'permutation_entropy',
               'lempel_ziv_complexity',
               'largest_lyapunov_exponent',
               ]
metric = plot_metrics[0]

#%% First we create a wide format frame from our data for plotting
metric_df1 = run1[run1['metric'] == metric]
metric_df1 = metric_df1.drop(labels=['label', 'metric', 'duration'], axis=1)
metric_df1 = metric_df1.set_index('startDataRecord')

#%% We can use this frame to show the development per channel over time
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(data=metric_df1, ax=ax)
fig.suptitle('Development of ' + metric + ' over time for individual channels')
fig.tight_layout()
plt.show()

#%% From this wide format frame we can convert to a long format frame for additional analysis
metric_df_long1 = metric_df1.reset_index() \
    .melt(id_vars='startDataRecord', var_name='channel', value_name='value') \
    .sort_values('startDataRecord', ascending=True)

#%% With this long format frame we can now do a very simple channel aggregation
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(data=metric_df_long1,x='startDataRecord', y='value', ax=ax)
fig.suptitle('Development of ' + metric + ' over time with channel std error')
fig.tight_layout()
plt.show()

#%% Now we apply the similar thing to the second run frame in order to be able to make comparisons between two result frames
metric_df2 = run2[run2['metric'] == metric]
metric_df2 = metric_df2.drop(labels=['label', 'metric', 'duration'], axis=1)
metric_df2 = metric_df2.set_index('startDataRecord')
metric_df_long2 = metric_df2.reset_index() \
    .melt(id_vars='startDataRecord', var_name='channel', value_name='value') \
    .sort_values('startDataRecord', ascending=True)
metric_df_long1['run'] = 'run1'
metric_df_long2['run'] = 'run2'
metric_df_longc = pd.concat([metric_df_long1, metric_df_long2])

#%% Now we have a long format frame which we can use to show the difference between the two
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(data=metric_df_longc,x='startDataRecord', y='value', hue='run', ax=ax)
fig.suptitle('Development of ' + metric + ' over time compared between runs')
fig.tight_layout()
plt.show()



