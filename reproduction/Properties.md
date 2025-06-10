# EEG data from external study
Data comes from a [different study](https://linkinghub.elsevier.com/retrieve/pii/S0960982215012427)
## participants
- 15 healthy adults
- 5 propfol, 5 xenon, 5 ketamine
## Recording
- atleast 5min resting wakefullness
- 3-5min of ansthesia (ramsey scale of 6 or lower)
- 60 channel EEG
- referenced to external electrode on the forehead
- online filter 0.1-350Hz
- fs 1450
- two eog channels
## Preprocessing
- offline 0.5-40Hz 3rd order butterworth with -3db at stopband (matlab filtfilt)
- bad channels rejected based on visual inspection (less than 10%)
- rejected eeg channels interpolated with spherical splines
- 2 second segments
- movement segments excluded based on visual inspection
- singular value decomposition to reduce the number of independent components to the number of good, non interpolated channels
- Ica using EEGLAB routines to remove ocular, muscle and electrocardiograph artifacts
- Data spans of 265+-64s (cleaned) EEG data available for every participant

# Metric calculation
## Avalance criticallity
### Prepare data for avalance detection
- z-transformed channels wise
- plotted in probability distribution spanning +- 10SD
- averaged distributions across all participants within same condition
- gaussian fit to each distribution
- binarization threshhold = point of divergence from distribution
- binarized: 0 for inside threshhold, 1 for outside
### Details of how avalanches are identified in binarized data
- multichannel analysis to find avalanches
- find 1 one in one channel: start of avalanche
- continue as long as there is atleast one channel with a 1
- end if condition is not met and start again at next 1
### Characteristics of avalanches
- Size (S): number of channels with a 1 during avalanche
- Duration (T): number of samples during avalanche
