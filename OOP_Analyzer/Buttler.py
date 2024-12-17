'''
This file contains the Buttler class. The buttler is a utility class which helps with reading file paths for example
'''

import os

class Buttler:
    def __init__(self):
        pass

    def check_outfile_name(self, outfile: str, file_exists_ok: bool = True):
        '''
        Checks if the output file already exists and if the provided file path is valid.
        
        Parameters:
        - outfile (str): The desired path where results should be saved. Must end with '_metrics.csv'.
        - file_exists_ok (bool): Indicates if it is acceptable for the file to already exist. Default is True.
        
        Returns:
        - tuple: 
          - (int): 1 if the file path is valid and the directory was created (if necessary), 
                   0 if the file name is invalid or does not meet the requirements.
          - (str): A message indicating the result of the operation.
        '''
        
        # Check if the output file name is non-empty and the path is valid
        if not outfile.strip():
            return 0, 'Output file path is empty or invalid.'
        
        # Check if the output file name ends with 'metrics.csv' and ensure that the directory exists
        if outfile.endswith('metrics.csv'):
            dirpath = os.path.dirname(outfile)
            os.makedirs(dirpath, exist_ok=file_exists_ok)
            return 1, 'Path was created, everything is OK.'
        else:
            return 0, 'Invalid file name. Ensure it ends with _metrics.csv.'


    def check_file_exists_and_create_path(self, log_file):
        '''
        Ensures that the directory path for the given log file exists. If it does not exist, it creates it.
    
        Parameters:
        - log_file (str): The full path to the log file, including its file name.
    
        Returns:
        - None
        '''
        directory = os.path.dirname(log_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def find_task_from_filename(self, filename):
        '''
        uses the filename to extract a task from it, essentially needs the keyword task in the name of the file and extracts
        '***_task-[extracts this]_***'
        input:
        -filename: name of the file to extract task from
        return:
        -task: name of the task that was extracted
        '''
        task: str = None
        file_name_list = filename.split('_')
        for name_part in file_name_list:
            if 'task' in name_part:
                task = name_part.split('-')[1:]
        if isinstance(task, list):
            task = '-'.join(task)
        return task


    def map_chaos_pipe_result_to_float(self, result: str) -> float:
        '''
        only relevant for the 0-1 chaos pipeline results from tokers original matlab implementation to convert a string to
        float
        '''
        # map the output of the chaos pipeline result to a float in order to have only floats in the dataframe
        if result == 'periodic':
            return 0
        if result == "chaotic":
            return 1
        if result == 'stochastic':
            return 2
        if result == 'nonstationary':
            return 3
        else:
            return 4  # something went wrong here
