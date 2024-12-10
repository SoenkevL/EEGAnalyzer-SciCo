'''
A file containing the Array_processor class. This class will take care of computing metrics on an array of data,
It will be able to use concepts such as sfreq, to convert between time and sample domain
'''


import numpy as np
import Metrics
import pandas as pd
from Buttler import Buttler

class Array_processor:
    def __init__(self, data=None, metric_name=None, sfreq=1, axis_of_time = 1):
        self.data = data if data else []
        self.metric_name = metric_name
        self.buttler = Buttler()
        self.sfreq = sfreq
        self.axis_of_time = axis_of_time

    def set_data(self, data):
        self.data = data

    def set_axis_of_time(self, axis_of_time):
        self.axis_of_time = axis_of_time

    def set_metric_name(self, metric_name):
        self.metric_name = metric_name

    def transpose_data(self, change_axis_of_time=False):
        self.data = self.data.T
        if change_axis_of_time:
            if self.axis_of_time == 1:
                self.axis_of_time = 0
            elif self.axis_of_time == 0:
                self.axis_of_time = 1

    def initialize_metric_functions(self, name):
        '''
        uses the Metrics initialized from Metrics.py and loads the functions, names, and kwargs for them
        inputs:
        -name: name of the metrics set that should be loaded
        returns:
        -metrics_functions: the list of functions to be calculated on the timeseries
        -metrics_name_list: list of names for the functions used to save the results
        -kwargs_list: list of dictionaries with additional arguments for the functions
        '''
        metrics_functions, metrics_name_list, kwargs_list = Metrics.select_metrics(name)
        return metrics_functions, metrics_name_list, kwargs_list


    def apply_metric_func(self, data, metric_func, kwargs):
        '''
        applies a function to a timeseries (data channel)
        inputs:
        -data: channel data (one dimensional time series)
        -metric_func: function which is calculated based on the data
        -kwargs: additional arguments for the function
        returns:
        -function ouput after calculation on the data
        '''
        # print(f'{datetime.today().strftime("%Y-%m-%d %H:%M:%S")} : Applying function {metric_func.__name__}'
        #       f' with hyperparameters: {kwargs}')

        # ensures eeg channel is saved as contiguos array in memory
        data = np.ascontiguousarray(data)
        if kwargs:
            try:
                # try applying kwargs as kwargs
                return metric_func(data, **kwargs)
            except TypeError:
                # if kwargs are not accepted use kwarg values as args list and try again
                return metric_func(data, *list(kwargs.values()))
            except Exception as e:
                print(f'could not apply metric {metric_func.__name__} to eeg with exception {e}')
        else:
            # if no kwargs are given just calculate with default parameters
            return metric_func(data)


    def create_result_array(self, eeg_np_array, metrics_func_list: list, kwargs_list: list[dict]) -> list:
        '''
        Creates a list of results using the function list and eeg data
        inputs:
        -eeg_np_array: array converted to numpy standard from the eeg frame
        -metrics_func_list: list of functions to compute on eeg
        -kwargs_list: list of additional arguments for the metrics
        returns:
        -result_list: list that contains the results per metric for the eeg data
        '''
        result_list = []
        for metric_func, kwargs in zip(metrics_func_list, kwargs_list):
            result_list.append(self.apply_metric_func(eeg_np_array, metric_func, kwargs))
        return result_list


    ############################################ advanced functions ########################################################


    def process_result_array(self, result_array: list, metric_name_array: list[str]) -> list:
        '''
        The results from the metrics are often returned as lists or dictionaries with additional parameters which we dont
        want to have in the csv
        inputs:
        results_array: array of results that can be of type dict or list
        metric_name_array: array of names for the metrics in the csv
        returns:
        processed_array: array with (name, metric) tuples and extracted from the original form
        '''
        # only keep the relevant metric parts for further processing
        processed_array = []
        for result in result_array:
            result_type = type(result)
            if result_type == list or result_type == tuple:
                processed_array.append(result[0])
            elif result_type == dict:
                for key, value in result.items():
                    if key == 'result':
                        value = self.buttler.map_chaos_pipe_result_to_float(value)
                    processed_array.append(value)
            else:
                processed_array.append(result)
        # create tuples with names
        for i, (name, pai) in enumerate(zip(metric_name_array, processed_array)):
            processed_array[i] = (name, pai)
        return processed_array


    def create_result_dict_from_eeg_frame(self, data_frame, metrics_func_list: list,
                                          metrics_name_list: list[str], kwargs_list: list[dict],
                                          channelwise=True) -> (dict, list[str]):
        '''
        Creates a metric frame from the combination of the eeg and the metric lists
        input:
        metrics_func_list: list of the metric functions
        metrics_name_list: list of names of the metric functions
        kwargs_list: list of dictionaries with additional arguments
        axis: along which axis metrics should be computed, 1 for columns, 0 for rows
        channelwise: if metrics should be computet per single timeseries or on the full frame
        returns:
        result_dict: dictionary with keys beeing the eeg columns and values the computed metrics
        metrics_name_list: list of the metric names
        '''
        result_dict = {}
        if channelwise:
            if self.axis_of_time == 0:
                for col in range(data_frame.shape[1]):
                    temp_data = data_frame[:, col]
                    raw_result_array = self.create_result_array(temp_data, metrics_func_list, kwargs_list)
                    processed_result_array = self.process_result_array(raw_result_array, metrics_name_list)
                    result_dict[col] = processed_result_array
            else:
                for row in range(data_frame.shape[0]):
                    temp_data = data_frame[row, :]
                    raw_result_array = self.create_result_array(temp_data, metrics_func_list, kwargs_list)
                    processed_result_array =self.process_result_array(raw_result_array, metrics_name_list)
                    result_dict[row] = processed_result_array
        else:
            raw_result_array = self.create_result_array(self.data, metrics_func_list, kwargs_list)
            processed_result_array = self.process_result_array(raw_result_array, metrics_name_list)
            result_dict = {column: processed_result_array for column in range(self.data.shape[1])}
            #pprint(result_dict, indent=4)
        return result_dict, metrics_name_list


    def create_dataframe_from_result_dict(self, result_dict: dict, metric_name_array: list[str],
                                          start_data_record: float, duration: float, label: str) -> pd.DataFrame:
        '''
        Takes in result dictionary with some additional parameters to create the dataframe
        input:
        result_dict: dictionary from create_result_dict_from_eeg_frame with eeg columns and the according values
        metric_name_array: array of names for the lists provided in the above dictionary
        start_data_record: float containing the start time of the eeg
        duration: duration of the eeg segment
        label: label of the eeg
        returns:
        sub_results_frame:
        '''
        # create a multiindexing based on metric, label, startDataRecord, duration
        index = pd.MultiIndex.from_product([[label], [start_data_record], [duration], metric_name_array],
                                           names=['label', 'startDataRecord', 'duration', 'metric'])
        eeg_column_names = list(result_dict.keys())
        sub_results_frame = pd.DataFrame(columns=eeg_column_names,
                                         index=index,
                                         dtype=float)
        # put the results into the dataframe
        for column, result_array in result_dict.items():
            for result_tuple in result_array:
                result = result_tuple[1]
                metric_name = result_tuple[0]
                sub_results_frame.loc[(label, start_data_record, duration, metric_name), column] = result
        return sub_results_frame


    def calc_metrics_from_eeg_dataframe_and_annotations(self, dataframe,
                                                        annot_label: str, annot_startDataRecord: float,
                                                        annot_duration: float) -> pd.DataFrame:
        '''
        Function which combines sub steps of computing metrics, processing the results and creating a sub dataframe from
        the metrics which can be later combined to a larger frame.
        input:
        - eeg_dataframe: pandas frame with eeg data to be analyzed
        - metric_set_name: name of the metric set
        - annot_label: label of the annotation of the eeg
        - annot_startDataRecord: time of the start of the annotation
        - annot_duration: duration of the annotated segment to be analyzed
        returns:
        - sub_results_frame: dataframe containing the metrics per channel for the segment
        '''
        # initialize the metrics which should be calculated
        metrics_functions, metrics_name_list, kwargs_list = self.initialize_metric_functions(self.metric_name)
        # calculate the results for the metrics and store them to dict
        result_dict, metrics_name_list = self.create_result_dict_from_eeg_frame(dataframe, metrics_functions,
                                                                           metrics_name_list, kwargs_list)
        # create the sub_results_frame from results dict
        sub_results_frame = self.create_dataframe_from_result_dict(result_dict, metrics_name_list, annot_startDataRecord,
                                                              annot_duration, annot_label)
        return sub_results_frame


    def epoching(self, duration: int = None, start_time: int = None, stop_time: int = None,
                 overlap: int = 0, task: str = None) -> pd.DataFrame:
        '''
        instead of using annotations in the data constant epochs are used
        duration, start_time and stop_time need to be given in seconds. Only duration is mandatory.
        epochs are created in the interval [start, stop) with steps of duration
        only full duration intervals are created
        '''
        n_samples = len(self.data)
        full_epoch_frame = pd.DataFrame()
        # set default parameters
        if not stop_time:
            stop_time = np.round(n_samples / self.sfreq)
        else:
            stop_time = int(min(np.round(n_samples / self.sfreq), stop_time))
        if not start_time:
            start_time = 0
        if not duration:
            duration = int(np.floor(stop_time - start_time))
        if overlap:
            if overlap >= duration:
                overlap = 0
                print('overlap cant be bigger or equal to the duration, was reset to 0')
        # go through the different epochs
        for t_onset in np.arange(start_time, (stop_time - duration) + 1, duration-overlap):
            t_onset_samples = t_onset * self.sfreq
            t_stop_samples = (t_onset + duration) * self.sfreq
            eeg_dataframe = self.data[[t_onset_samples, t_stop_samples], :]
            # calculat the results of the current epoch and save them to dataframe
            print(f'Calculating for times: {t_onset}:{t_onset + duration}')
            sub_results_frame = self.calc_metrics_from_eeg_dataframe_and_annotations(eeg_dataframe,
                                                                                task, t_onset, duration)
            # add the calculated metrics to the frame with all the other epochs
            full_epoch_frame = pd.concat([full_epoch_frame, sub_results_frame], axis=0)
        return full_epoch_frame