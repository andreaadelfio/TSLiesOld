'''
Utils module for the ACNBkg project.
'''
import sys
import os
import pprint
import re
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import fftpack
try:
    from modules.config import INPUTS_OUTPUTS_FILE_PATH, LOGGING_FILE_PATH, INPUTS_OUTPUTS_FOLDER, DIR, DATA_LATACD_FOLDER_NAME
except:
    from config import INPUTS_OUTPUTS_FILE_PATH, LOGGING_FILE_PATH, INPUTS_OUTPUTS_FOLDER, DIR, DATA_LATACD_FOLDER_NAME

class Logger():
    '''
    A class that provides utility functions for logging.

    '''
    def __init__(self, logger_name: str,
                 log_file_name: str = LOGGING_FILE_PATH,
                 log_level: int = logging.DEBUG):
        '''
        Initializes a Logger object.

        Parameters:
        ----------
            logger_name (str): The name of the logger.
            log_file_name (str): The name of the log file. 
                                 Default is LOGGING_FILE_PATH from config.py.
            log_level (int): The log level. Default is logging.DEBUG.

        Returns:
        -------
            None
        '''
        if not os.path.exists(log_file_name):
            os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        self.log_file_name = log_file_name
        self.log_level = log_level
        self.logger_name = logger_name
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.log_level)
        self.format = '%(asctime)s %(name)s [%(levelname)s]: %(pathname)s - %(funcName)s : %(message)s'
        self.formatter = logging.Formatter(self.format)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.file_handler = logging.FileHandler(self.log_file_name)
        self.file_handler.setLevel(self.log_level)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def get_logger(self):
        '''
        Returns the logger object.

        Returns:
        -------
            logging.Logger: The logger object.
        '''
        return self.logger

def logger_decorator(logger):
    '''Logger decorator'''
    def decorator(func):
        def wrapper(*args, **kwargs):
            class CustomLogRecord(logging.LogRecord):
                '''Custom Log Record, changes pathname and funcName'''
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.pathname = sys.modules.get(func.__module__).__file__
                    self.funcName = func.__name__
            logging.setLogRecordFactory(CustomLogRecord)
            logger.info('START')
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logging.setLogRecordFactory(CustomLogRecord)
                logger.error(e)
                logger.debug('Args: %s', pprint.pformat(args))
                logger.debug('Kwargs: %s', pprint.pformat(kwargs))
                raise
            logging.setLogRecordFactory(CustomLogRecord)
            logger.info('END')
            return result
        return wrapper
    return decorator

class Time:
    '''Class to handle time datatype
    '''
    logger = Logger('Time').get_logger()

    fermi_ref_time = datetime(2001, 1, 1, 0, 0, 0)
    fermi_launch_time = datetime(2008, 8, 7, 3, 35, 44)

    @staticmethod
    def from_met_to_datetime(met_list: list) -> list:
        '''
        Convert the MET to a datetime object.

        Parameters:
        - met_list (list): The MET list to convert.

        Returns:
        -------
        - datetime_list (list of datetime): The datetime object corresponding to the MET.
        '''
        return [Time.fermi_ref_time + timedelta(seconds=int(met)) for met in met_list]

    @staticmethod
    def from_datetime_to_met(datetime_list: list) -> list:
        '''
        Convert the datetime object to MET.

        Parameters:
        - datetime_list (list): The datetime list to convert.

        Returns:
        -------
        - met_list (list of int): The MET corresponding to the datetime object.
        '''
        for dt in datetime_list:
            print(dt)
        return [(dt - Time.fermi_ref_time).total_seconds() for dt in datetime_list]

    @staticmethod
    def from_met_to_datetime_str(met_list: list) -> list:
        '''
        Convert the MET to a datetime object and return as string.

        Parameters:
        - met_list (list): The MET list to convert.

        Returns:
        -------
        - datetime_list (list of str): The datetime object corresponding to the MET, 
                                       represented as strings.
        '''
        return [str(Time.fermi_ref_time + timedelta(seconds=int(met))) for met in met_list]

    @staticmethod
    def remove_milliseconds_from_datetime(datetime_list: list) -> list:
        '''
        Remove the milliseconds from the datetime object.

        Parameters:
        - datetime_list (list): The datetime list to convert.

        Returns:
        -------
        - datetime_list (list of datetime): The datetime object without milliseconds.
        '''
        return [dt.replace(microsecond=0) for dt in datetime_list]

    @staticmethod
    def get_week_from_datetime(datetime_dict: dict) -> list:
        '''
        Get the week number from the datetime object.

        Parameters:
        - datetime_dict (list): The datetime list to convert.

        Returns:
        -------
        - week_list (list of int): The week number corresponding to the datetime.
        '''
        weeks_set = set()
        for dt1, dt2 in datetime_dict.values():
            weeks_set.add(((dt1 - Time.fermi_launch_time).days) // 7 + 10)
            weeks_set.add(((dt2 - Time.fermi_launch_time).days) // 7 + 10)
        return list(range(min(weeks_set), max(weeks_set) + 1))

    @staticmethod
    def get_datetime_from_week(week: int) -> tuple:
        '''
        Get the datetime from the week number.

        Parameters:
        - week (int): The week number.

        Returns:
        -------
        - datetime_tuple (tuple): The datetime tuple corresponding to the week.
        '''
        start = Time.fermi_launch_time + timedelta(weeks=week - 10)
        end = start + timedelta(weeks=1)
        return start, end

class Data():
    '''Class to handle data
    '''
    logger = Logger('Data').get_logger()
    '''
    A class that provides utility functions for data manipulation.
    '''

    @logger_decorator(logger)
    @staticmethod
    def get_masked_dataframe(start, stop, data, column='datetime', reset_index=False) -> pd.DataFrame:
        '''
        Returns the masked data within the specified time range.

        Parameters:
        ----------
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
            data (DataFrame): The input data.
            column (str, optional): The column name representing the time. Defaults to 'datetime'.

        Returns:
        -------
            DataFrame: The masked data within the specified time range.
        '''
        if column == 'index':
            mask = (data.index >= start) & (data.index <= stop)
        else:
            mask = (data[column] >= start) & (data[column] <= stop)
        masked_data = data[mask].reset_index(drop=True) if reset_index else data[mask]
        return pd.DataFrame(masked_data)

    @staticmethod
    def get_excluded_dataframes(start, stop, data, column='datetime'):
        '''
        Returns the excluded dataframes within the specified time range.

        Parameters:
        ----------
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
            data (DataFrame): The input data.
            column (str, optional): The column name representing the time. Defaults to 'datetime'.

        Returns:
        -------
            list: The excluded dataframes within the specified time range.
        '''
        mask = (data[column] < start) | (data[column] > stop)
        excluded_data = data[mask]
        return excluded_data

    @staticmethod
    def get_masked_data(start, stop, data, column='datetime'):
        '''
        Returns the masked data within the specified time range.

        Parameters:
        ----------
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
            data (DataFrame): The input data.
            column (str, optional): The column name representing the time. Defaults to 'datetime'.

        Returns:
        -------
            dict: The masked data within the specified time range, with column names as keys
                  and lists of values as values.
        '''
        mask = (data[column] >= start) & (data[column] <= stop)
        masked_data = data[mask]
        return {name: masked_data.field(name).tolist() for name in masked_data.names}

    @logger_decorator(logger)
    @staticmethod
    def filter_dataframe_with_run_times(initial_dataframe, run_times):
        '''
        Returns the spacecraft dataframe filtered on runs times.

        Parameters:
        ----------
            initial_dataframe (DataFrame): The initial spacecraft data.
            run_times (DataFrame): The dataframe containing run times.

        Returns:
        -------
            DataFrame: The filtered spacecraft dataframe.
        '''
        df = pd.DataFrame()
        for start, end in run_times.values():
            df = pd.concat([df, Data.get_masked_dataframe(data=initial_dataframe,
                                                          start=start, stop=end)],
                            ignore_index=True)
        return df

    @logger_decorator(logger)
    @staticmethod
    def convert_to_df(data_to_df: np.ndarray) -> pd.DataFrame:
        '''
        Converts the data containing the spacecraft data into a pd.DataFrame.

        Parameters:
        ----------
            data_to_df (ndarray): The data to convert.

        Returns:
        -------
            DataFrame: The dataframe containing the spacecraft data.
        '''
        name_data_dict = {name: data_to_df.field(name).tolist() for name in data_to_df.dtype.names}
        return pd.DataFrame(name_data_dict)

    @logger_decorator(logger)
    @staticmethod
    def merge_dfs(first_dataframe: pd.DataFrame, second_dataframe: pd.DataFrame,
                  on_column='datetime') -> pd.DataFrame:
        '''
        Merges two dataframes based on a common column.

        Parameters:
        ----------
            first_dataframe (DataFrame): The first dataframe.
            second_dataframe (DataFrame): The second dataframe.
            on_column (str, optional): The column to merge on. Defaults to 'datetime'.

        Returns:
        -------
            DataFrame: The merged dataframe.
        '''
        return pd.merge(first_dataframe, second_dataframe, on=on_column, how='inner')

class File:
    '''Class to handle files
    '''
    logger = Logger('File').get_logger()

    @logger_decorator(logger)
    @staticmethod
    def write_df_on_file(df: pd.DataFrame, filename: str=INPUTS_OUTPUTS_FILE_PATH, fmt: str='pk'):
        '''
        Write the dataframe to a file.

        Parameters:
        ----------
            df (DataFrame): The dataframe to write.
            filename (str, optional): The name of the file to write the dataframe to.
                                      Defaults to INPUTS_OUTPUTS_FILE_PATH.
            fmt (str, optional): The format to write the dataframe in.
                                 Can be:
                                    'csv' to write a .csv file;
                                    'pk' to write a .pk file;
                                    'both' to write in both formats.
                                 Defaults to 'pk'.
        '''
        path, filename = os.path.split(filename)
        if fmt == 'csv':
            if not os.path.exists(os.path.join(path, 'csv')):
                os.makedirs(os.path.join(path, 'csv'), exist_ok=True)
            df.to_csv(os.path.join(path, 'csv', filename + '.csv'), index=False)
        elif fmt == 'pk':
            if not os.path.exists(os.path.join(path, 'pk')):
                os.makedirs(os.path.join(path, 'pk'), exist_ok=True)
            df.to_pickle(os.path.join(path, 'pk', filename + '.pk'))
        elif fmt == 'both':
            if not os.path.exists(os.path.join(path, 'csv')):
                os.makedirs(os.path.join(path, 'csv'), exist_ok=True)
            if not os.path.exists(os.path.join(path, 'pk')):
                os.makedirs(os.path.join(path, 'pk'), exist_ok=True)
            df.to_csv(os.path.join(path, 'csv', filename + '.csv'), index=False)
            df.to_pickle(os.path.join(path, 'pk', filename + '.pk'))

    @logger_decorator(logger)
    @staticmethod
    def read_df_from_file(filename=INPUTS_OUTPUTS_FILE_PATH):
        '''
        Read the dataframe from a file.

        Parameters:
        ----------
            filename (str, optional): The name of the file to read the dataframe from.
                                      Defaults to INPUTS_OUTPUTS_FILE_PATH.

        Returns:
        -------
            DataFrame: The dataframe read from the file.
        '''
        path = os.path.join(DIR, f'{filename}.pk')
        if os.path.exists(path):
            return pd.read_pickle(path)
        return None
    
    @staticmethod
    def add_smoothing_with_mean(tile_signal, window=30):
        '''This function adds the smoothed histograms to the signal dataframe.'''
        window = window if len(tile_signal) > window else len(tile_signal)
        for h_name in set(tile_signal.keys()) - {'MET', 'datetime'}:
            histc = tile_signal[h_name].to_list()
            filtered_sig1 = np.convolve(histc, np.ones(window)/window, mode='same')
            for i in range(window//2):
                filtered_sig1[i] = np.mean(histc[:i+window//2])
            for i in range(len(histc) - window//2, len(histc)):
                filtered_sig1[i] = np.mean(histc[i-window//2:])
            tile_signal[f'{h_name}_smooth'] = filtered_sig1
        return tile_signal

    @staticmethod
    def add_smoothing_with_fft(tile_signal):
        '''This function adds the smoothed histograms to the signal dataframe.'''
        histx = tile_signal['MET']
        time_step = histx[2] - histx[1]
        nyquist_freq = 0.5 / time_step
        for h_name in set(tile_signal.keys()) - {'MET', 'datetime'}:
            histc = tile_signal[h_name].to_list()
            freq_cut1 = np.mean(histc) * nyquist_freq / 3
            sig_fft = fftpack.fft(histc)
            sample_freq = fftpack.fftfreq(len(histc), d=time_step)
            low_freq_fft1  = sig_fft.copy()
            low_freq_fft1[np.abs(sample_freq) > freq_cut1] = 0
            filtered_sig1  = np.array(fftpack.ifft(low_freq_fft1)).real
            tile_signal[f'{h_name}_smooth'] = filtered_sig1
        return tile_signal

    @logger_decorator(logger)
    @staticmethod
    def read_dfs_from_runs_pk_folder(folder_path=INPUTS_OUTPUTS_FOLDER, add_smoothing=False, mode='mean', window=30):
        '''
        Read the dataframe from pickle files in a folder.

        Parameters:
        ----------
            folder_path (str, optional): The name of the folder to read the dataframe from.
                                      Defaults to INPUTS_OUTPUTS_FOLDER.

        Returns:
        -------
            DataFrame: The dataframe read from the file.
        '''
        folder_path = os.path.join(folder_path, 'pk')
        merged_dfs: pd.DataFrame = None
        if os.path.exists(folder_path):
            dir_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                        if file.endswith('.pk')]
            dir_list = sorted(dir_list, key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group(0)))
            dfs = [pd.read_pickle(file) for file in dir_list]
            dfs = [df[(df.loc[:, df.columns.difference(['MET'])] != 0).any(axis=1)] for df in dfs]
            if add_smoothing:
                if mode == 'mean':
                    dfs = [File.add_smoothing_with_mean(df, window=window) for df in dfs]
                elif mode == 'fft':
                    dfs = [File.add_smoothing_with_fft(df) for df in dfs]
            merged_dfs = pd.concat(dfs, ignore_index=True).drop_duplicates('MET', ignore_index=True) # patch, trovare sorgente del bug
        return merged_dfs

    @logger_decorator(logger)
    @staticmethod
    def read_dfs_from_weekly_pk_folder(folder_path=INPUTS_OUTPUTS_FOLDER, custom_sorter=lambda x: int(x.split('_w')[-1].split('.')[0]), cols_list=None, start=None, stop=None):
        '''
        Read the dataframe from pickle files in a folder.

        Parameters:
        ----------
            folder_path (str, optional): The name of the folder to read the dataframe from.
                                      Defaults to INPUTS_OUTPUTS_FOLDER.

        Returns:
        -------
            DataFrame: The dataframe read from the file.
        '''
        folder_path = os.path.join(folder_path, 'pk')
        merged_dfs: pd.DataFrame = None
        if os.path.exists(folder_path):
            dir_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                        if file.endswith('.pk') and start <= int(file.split('_w')[-1].split('.')[0]) <= stop]
            dir_list = sorted(dir_list, key=custom_sorter)
            dfs = [pd.read_pickle(file)[cols_list] if cols_list else pd.read_pickle(file) for file in dir_list]
            merged_dfs = pd.concat(dfs, ignore_index=True).drop_duplicates('MET', ignore_index=True) # patch, trovare sorgente del bug
        return merged_dfs

    @logger_decorator(logger)
    @staticmethod
    def read_dfs_from_csv_folder(folder_path=INPUTS_OUTPUTS_FOLDER, custom_sorter=lambda x: int(x.split('_w')[-1].split('.')[0])):
        '''
        Read the dataframe from csv files in a folder.

        Parameters:
        ----------
            folder_path (str, optional): The name of the folder to read the dataframe from.
                                      Defaults to INPUTS_OUTPUTS_FOLDER.

        Returns:
        -------
            DataFrame: The dataframe read from the file.
        '''
        folder_path = os.path.join(folder_path, 'csv')
        if os.path.exists(folder_path):
            dir_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
            dir_list = sorted(dir_list, key=custom_sorter)
            dfs = [pd.read_csv(file) for file in dir_list]
            merged_dfs = pd.concat(dfs, ignore_index=True).drop_duplicates('MET', ignore_index=True) # patch, trovare sorgente del bug
        return merged_dfs

    @logger_decorator(logger)
    @staticmethod
    def write_on_file(data: dict, filename: str):
        '''Writes data on file

        Parameters:
        ----------
            data (dict): disctionary containing data
            filename (str): name of the file
        '''
        with open(filename, 'w', encoding='utf-8') as file:
            for key, value in data.items():
                file.write(f'{key}: {value}\n')

    @logger_decorator(logger)
    @staticmethod
    def check_integrity_runs_pk_folder(folder_path=INPUTS_OUTPUTS_FOLDER):
        '''
        Checks the integrity of the dataframe from pickle files in a folder.

        Parameters:
        ----------
            folder_path (str, optional): The name of the folder to read the dataframe from.
                                      Defaults to INPUTS_OUTPUTS_FOLDER.
        '''
        print('Checking integrity...')
        folder_path = os.path.join(folder_path, 'pk')
        if os.path.exists(folder_path):
            dir_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                        if file.endswith('.pk')]
            dir_list = sorted(dir_list, key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group(0)))
            for file in dir_list:
                df = pd.read_pickle(file)
                initial_len = len(df)
                df = df[(df.loc[:, df.columns.difference(['MET', 'datetime'])] != 0).any(axis=1)]
                if len(df) != initial_len:
                    print(initial_len, len(df), os.path.basename(file))
                    # os.remove(file)

if __name__ == '__main__':
    File.check_integrity_runs_pk_folder(DATA_LATACD_FOLDER_NAME)