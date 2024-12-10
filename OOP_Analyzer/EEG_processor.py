'''
This file contains the EEG_processor which takes as input a file to an eeg which is readable by eeg e.g. edf, fif
Can do loading, preprocessing and exporting of eeg data for further processing
'''

import pandas as pd
import icecream as ic
import mne

from OOP_Analyzer.Array_processor import Array_processor


class EEG_processor:
    def __init__(self, datapath):
        self.datapath = datapath
        self.raw, self.sfreq = self.load_data_file(self.datapath)
        self.info = self.raw.info
        self.Array_processor = Array_processor(sfreq=self.sfreq)


    def load_data_file(data_file: str):
        '''
        Uses mne to load a readable file format into a raw instance and extracts its sampling frequency
        input:
        - data_file: datapath of the file you want to load (required)
        outputs:
        - raw: raw instance from mne
        - sfreq: sampling frequency of the EEG
        '''
        try:
            raw = mne.io.read_raw(data_file, preload=True)
            sfreq = raw.info['sfreq']
            return raw, sfreq
        except Exception as e:
            print(f'!!!!!!!!!!!!!!!!! could not load file see error below !!!!!!!!!!!!!!!!!!!!!  \n {e}')
            return None, None

    def load_data_of_raw_object(self):
        self.raw.load_data()

    def apply_filter(self, l_freq: float = None, h_freq: float = None, picks: str = 'eeg'):
        '''
        filters a raw instance and returns it afterwards, can be used for lowpass, highpass, bandpass, bandstop filter
        inputs:
        -raw: raw instance (required)
        -l_freq: the lower end of frequencies that should be passed, so essentially only frequencies above it will be included
        -h_freq: the higher end of frequencies that should be passed, so essentially only frequencies below it will be included
        -picks: makes sure only eeg channels are used which can be helpful from time to time
        outputs:
        -None, instead alters the inputed raw instance
        '''
        if l_freq and l_freq != 'None' and h_freq and h_freq != 'None':
            self.raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks)
        elif l_freq and l_freq != 'None':
            self.raw.filter(l_freq=l_freq, h_freq=self.raw.info['sfreq'], picks=picks)
        elif h_freq and h_freq != 'None':
            self.raw.filter(l_freq=0, h_freq=l_freq, picks=picks)


    def only_keep_10_20_channels_and_check_bipolar(self):
        '''
        Checks the EEG channels for containing a valid part, only once and no part that is marked as invalid in order to
        only keep the correct EEG channels and marke others as bad
        returns True if two valid channel parts are in one channel
        inputs:
        -raw: raw instance from EEG
        outputs:
        -duplicate_positive: returns True if a bipolar montage is used as the channel names contain two channels
        '''
        valid_channel_parts = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                               'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
        invalid_channel_parts = ['pO2', 'CO2', 'X', 'Res', 'time']
        added_bads = []
        duplicate_positive = False
        for ch in self.raw.ch_names:
            bad_channel = True
            for vcp in valid_channel_parts:
                if vcp.lower() in ch.lower():
                    if bad_channel:
                        bad_channel = False
                    else:
                        duplicate_positive = True
            for icp in invalid_channel_parts:
                if icp.lower() in ch.lower():
                    bad_channel = True
            if bad_channel:
                added_bads.append(ch)
        self.self.raw.info['bads'] = self.raw.info['bads'] + added_bads
        self.raw.info['bads'] = list(set(self.raw.info['bads']))
        return duplicate_positive


    def convert_electrode_names_to_channel_names(self, electrode_names: list[str], channel_names: list[str]):
        '''
        Goes through the electrode names and converts them to channel names if the electrode name is part of the channel name
        inputs:
        -electrode_names: names of the electrodes from the montage
        -channel_names: names of the channels after conversion to ensure unified names in the outputs
        returns:
        -outputs_array: array of the same size as electrode_names containing either None or the according channel name
        '''
        output_array = list()
        for i, e_name in enumerate(electrode_names):
            output_array.append(None)
            for c_name in channel_names:
                if e_name.lower() in c_name.lower():
                    output_array[i] = c_name
                    break
        return output_array


    def ensure_electrodes_present(self, anodes, cathods, new_names):
        '''
        checks if the anode and cathode for the bipolar reference are present, if not they will be dropped
        inputs:
        -anodes: a list of anode names which contains the name or None if the electrode is not present
        -cathodes: a list of cathode names which contains the name or None if the electrode is not present
        -new_names: a list of the bipolar channel names
        returns:
        -droped_names: all bipolar channels for which stuff was missing
        -new_names: new names which did not have to be dropped
        -anodes: anodes which have not been dropped
        -cathodes: cathodes which have not been dropped
        '''
        drop_idx = []
        for i, (a, c, nn) in enumerate(zip(anodes, cathods, new_names)):
            if a and c:
                continue
            else:
                drop_idx.append(i)
        droped_names = [dn for i, dn in enumerate(new_names) if i in drop_idx]
        new_names = [nn for i, nn in enumerate(new_names) if i not in drop_idx]
        anodes = [a for i, a in enumerate(anodes) if i not in drop_idx]
        cathods = [c for i, c in enumerate(cathods) if i not in drop_idx]
        return anodes, cathods, new_names, droped_names



    def change_montage(self, montage: str):
        '''
        changes the montage of a raw instance
        input:
        -raw: raw instance of the EEG
        -montage: name of the montage, either 'avg', 'doublebanana', 'circumferential' or a specific channel
        returns:
        -raw_internal: the changed raw instance, can be None if there is an error during computation
        '''
        # in this step now automatically some channels are excluded if marked as bad or not marked as EEG channels
        raw_internal = self.raw.pick(exclude='bads', picks='eeg').copy()
        # change the reference used in the eeg file to avg
        if montage == 'avg':
            ic(raw_internal.set_eeg_reference(ref_channels='average'))
        # change montage to doublebanana
        elif montage == 'doublebanana':
            anodes = ['Fp2', 'F8', 'T4', 'T6',
                      'Fp2', 'F4', 'C4', 'P4',
                      'Fz', 'Cz',
                      'Fp1', 'F3', 'C3', 'P3',
                      'Fp1', 'F7', 'T3', 'T5']
            cathodes = ['F8', 'T4', 'T6', 'O2',
                        'F4', 'C4', 'P4', 'O2',
                        'Cz', 'Pz',
                        'F3', 'C3', 'P3', 'O1',
                        'F7', 'T3', 'T5', 'O1']
            new_names = ['Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
                         'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                         'Fz-Cz', 'Cz-Pz',
                         'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                         'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1']
            # make sure names are unified and match the above
            anode_eeg_channels = self.convert_electrode_names_to_channel_names(anodes, raw_internal.ch_names)
            cathode_eeg_channels = self.convert_electrode_names_to_channel_names(cathodes, raw_internal.ch_names)
            try:
                # try setting the new bipolar reference
                ic(mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                             ch_name=new_names, copy=False))
            except ValueError:
                try:
                    # if setting failes in the previous step we need to check all channels are present and match
                    anode_eeg_channels, cathode_eeg_channels, new_names, dropped_names = self.ensure_electrodes_present(
                        anode_eeg_channels, cathode_eeg_channels, new_names)
                    print(
                        f'montage could not be set fully, probably not all needed channels are present. The following channels could not be computed: {dropped_names}')
                    ic(mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                                 ch_name=new_names, copy=False))
                except ValueError:
                    print('montage could not be set at all')
                    return None
        # set cirumferential montage
        elif montage == 'circumferential':
            anodes = ['Fp2', 'F8', 'T4', 'T6',
                      'O2', 'O1', 'T5', 'T3',
                      'F7', 'Fp1']
            cathodes = ['F8', 'T4', 'T6',
                        'O2', 'O1', 'T5', 'T3',
                        'F7', 'Fp1', 'Fp2']
            new_names = ['Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
                         'O2-O1', 'O1-T5', 'T5-T3', 'T3-F7',
                         'F7-Fp1', 'Fp1-Fp2']
            # unify names
            anode_eeg_channels = self.convert_electrode_names_to_channel_names(anodes, raw_internal.ch_names)
            cathode_eeg_channels = self.convert_electrode_names_to_channel_names(cathodes, raw_internal.ch_names)
            try:
                # set montage
                mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                          ch_name=new_names, copy=False)
            except ValueError:
                # drop channels if montage could not be dropped
                anode_eeg_channels, cathode_eeg_channels, new_names, dropped_names = self.ensure_electrodes_present(
                    anode_eeg_channels, cathode_eeg_channels, new_names)
                print(
                    f'montage could not be set fully, probably not all needed channels are present. The following channels could not be computed: {dropped_names}')
                mne.set_bipolar_reference(inst=raw_internal, anode=anode_eeg_channels, cathode=cathode_eeg_channels,
                                          ch_name=new_names, copy=False)
        # set montage to a specific channel
        elif montage in raw_internal.ch_names:
            raw_internal.set_eeg_reference(ref_channels=montage)
        else:
            print(f'The given montage is not a viable option or a channel of the raw_internal object, no montage applied')
        return raw_internal


    def extract_eeg_columns(self, eeg_dataframe):
        '''
        Extracts all besides the first column from the eeg dataframe
        inputs:
        -dataframe with eeg data
        returns:
        -list of channels
        '''
        return eeg_dataframe.columns[1:]


    def calc_metric_from_annotations(self, metric_set_name, ep_dur: int, ep_start: int, ep_stop: int,
                                     overlap: int = 0, relevant_annot_labels: list = None) -> pd.DataFrame:
        '''
        calculates a dataframe for the full annotation epoched into segments.
        inputs:
        - raw: the raw object of the eeg
        - metrics_set_name: name of the metric set
        - ep_dur: duration of the epoch
        - ep_start: start of the epoch relative to eeg/annotation
        - ep_stop: maximum duration of the analyzed segment
        - relevant_annot_labels: list of annotations that should be analyzed, if None whole eeg is used
        returns:
        - full_annot_frame: pandas dataframe with metrics per channel for all epochs in an eeg/annotated eeg segment
        '''
        self.raw.load_data()
        data = self.raw.to_data_frame()
        Array_processor.set_data(data)
        if ep_start is None:
            ep_start = 0
        raw_annots = self.raw.annotations
        full_annot_frame = pd.DataFrame()
        # chec that the file actually has annotations
        if raw_annots:
            for annot in raw_annots:
                annot_name = annot['description']
                # if relevant annot labels are given select for them
                if relevant_annot_labels:
                    if annot_name in relevant_annot_labels:
                        # extract start and duration of annotation
                        annot_start_seconds = annot['onset']
                        annot_duration_seconds = annot['duration']
                        annot_stop_seconds = annot_start_seconds + annot_duration_seconds
                        print(f'annotation: {annot_name}, times: {annot_start_seconds}:{annot_stop_seconds}')
                        # use the epoching info to calculate the final times
                        ep_start_seconds = annot_start_seconds + ep_start
                        if ep_stop:
                            ep_stop_seconds = min(ep_start_seconds + ep_stop, annot_stop_seconds)
                        else:
                            ep_stop_seconds = annot_stop_seconds
                        # call the epoching function to calc the metrics
                        sub_results_frame = self.Array_processor.epoching(metric_set_name, ep_dur, ep_start_seconds, ep_stop_seconds,
                                                          overlap, annot_name)
                        # add the calculated metrics to the frame with all the annotations
                        full_annot_frame = pd.concat([full_annot_frame, sub_results_frame], axis=0)

                # if no relevant annot labels are given use all annots
                else:
                    # extract start and duration of annotation
                    annot_start_seconds = annot['onset']
                    annot_duration_seconds = annot['duration']
                    annot_stop_seconds = annot_start_seconds + annot_duration_seconds
                    # use the epoching info to calculate the final times
                    ep_start_seconds = annot_start_seconds + ep_start
                    ep_stop_seconds = min(ep_start_seconds + ep_stop, annot_stop_seconds)
                    # call the epoching function to calc the metrics
                    sub_results_frame = epoching(self.samples, metric_set_name, ep_dur, ep_start_seconds, ep_stop_seconds,
                                                 overlap, annot_name)
                    # add the calculated metrics to the frame with all the annotations
                    full_annot_frame = pd.concat([full_annot_frame, sub_results_frame], axis=0)
        return full_annot_frame


    def compute_metrics_fif(self, raw: mne.io.Raw, metric_set_name, relevant_annot_labels: list = None,
                            ep_dur=None, ep_start=None, ep_stop=None, overlap : int  = 0,
                            task_label=None) -> pd.DataFrame:
        '''
        wrapper functions around calc_metric_from_annotation or epoching to deal with eegs with one, multiple or no annotation
        inputs:
        - raw: the raw object of the eeg
        - metrics_set_name: name of the metric set
        - ep_dur: duration of the epoch
        - ep_start: start of the epoch relative to eeg/annotation
        - ep_stop: maximum duration of the analyzed segment
        - relevant_annot_labels: list of annotations that should be analyzed, if None whole eeg is used
        returns:
        - full_result_frame: pandas dataframe with metrics per channel for all epochs in an eeg/annotated eeg segment
        '''
        # annotation labels provided by user
        if relevant_annot_labels:
            # all annotations should be used
            if relevant_annot_labels[0] == 'all':
                full_results_frame = calc_metric_from_annotations(raw, metric_set_name, ep_dur, ep_start, ep_stop, None)
            # only the annotations in the relevant_annot_labels should be used
            else:
                full_results_frame = calc_metric_from_annotations(raw, metric_set_name, ep_dur, ep_start, ep_stop,
                                                                  relevant_annot_labels)
        # no annotation labels provided by user, whole file will be used
        else:
            full_results_frame = epoching(raw, metric_set_name, ep_dur, ep_start, ep_stop, overlap, task_label)
        return full_results_frame
