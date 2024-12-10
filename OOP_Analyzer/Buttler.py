'''
This file contains the Buttler class. The buttler is a utility class which helps with reading file paths for example
'''

import os

class Buttler:
    def __init__(self):
        pass


    def check_outfile_name(self, outfile: str, file_exists_ok: bool = True):
        '''
        utility function which checks if the output file allready exists and if the path is valid
        inputs:
        -outfile: path where results should be saved
        -file_exists_ok: if the file can allready exists or if that should give an error
        returns:
        -1 if everything with the file path is ok and 0 if there is a problem with the path
        '''
        # check that outputfile is correct and ensure that the path exists
        if outfile.endswith('metrics.csv'):
            dirpath = os.path.dirname(outfile)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=file_exists_ok)
                return 1, 'path was created, everything ok'
        else:
            return 0, 'name not valid, check if it ends with _metrics.csv'


    def check_file_exists_and_create_path(self,log_file):
        '''
        checks if a file path exists (not the file itself) and makes sure the path is created
        inputs:
        - log_file: the logfile name with path
        returns:
        - Nothing
        '''
        if os.path.dirname(log_file) and not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
