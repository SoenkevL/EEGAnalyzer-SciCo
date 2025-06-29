sub 005 awake has high emg power in A1, A2 channel which is why they were excluded
sub 006 anes has very high high frequency content in channel Fp2 -> not relevant within target region, no exclusion
sub 009 awake removed A1, A2 due to excessive high amplitude noise most probably due to high EMGsub 010 removed A1, A2 due to excessive high amplitude noise most probably due to high EMG
sub 010 awake, T1 shows higher emg noise but was not removed, maybe interpolation could be thought about
sub 011 awake, T3 shows high emg noise but was not removed, maybe interpolation could be thought about
### In sub 12 I stopped upsampling for computational reasons, as I can just upsample the relevant sections afterwards
- just have to be careful as now the ica is fitted on a lower frequency signal and less datapoints on average which could have an influence
sub 21 is beeing excluded as it has no marked eeg section for beeing anesthesized and apparently also received fentanyl not only propofol
I could not find a clear difference in ICA decomposition quality after not upsampling anymore, therfore I assume that was a valid decision
- Ofcourse still possible to redo the first EEGs until 12 without the upsampling
Finished EEG preprocessing

Next: Write a quick script which analysis how much seconds of anes, awake I have per patient for the descriptive statistics