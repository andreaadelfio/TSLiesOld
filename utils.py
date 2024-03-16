"""Utils module for the ACNBkg project."""
import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import pandas as pd
from config import INPUTS_OUTPUTS_FILE_PATH, LOGGING_FILE_PATH


class Logger():
    """
    A class that provides utility functions for logging.

    """
    def __init__(self, logger_name: str,
                 log_file_name: str = LOGGING_FILE_PATH,
                 log_level: int = logging.DEBUG):
        '''
        Initializes a Logger object.

        Args:
            logger_name (str): The name of the logger.
            log_file_name (str): The name of the log file. Default is LOGGING_FILE_PATH from config.py.
            log_level (int): The log level. Default is logging.DEBUG.

        Returns:
            None
        '''
        if not os.path.exists(log_file_name):
            os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        self.log_file_name = log_file_name
        self.log_level = log_level
        self.logger_name = logger_name
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.log_level)
        self.format = '%(asctime)s %(name)s [%(levelname)s]: %(pathname)s - %(funcName)s (%(lineno)d) : %(message)s'
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
            logging.Logger: The logger object.
        '''
        return self.logger


def logger_decorator(logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            class CustomLogRecord(logging.LogRecord):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.pathname = sys.modules.get(func.__module__).__file__
                    self.funcName = func.__name__
            logging.setLogRecordFactory(CustomLogRecord)
            logger.info(f'{func.__name__} - START')
            result = func(*args, **kwargs)
            logger.info(f'{func.__name__} - END')
            return result
        return wrapper
    return decorator

class Time:
    logger = Logger('Time').get_logger()

    fermi_ref_time = datetime(2001, 1, 1, 0, 0, 0)


    def from_met_to_datetime(met_list) -> list:
        """
        Convert the MET to a datetime object.

        Parameters:
        - met_list (list): The MET list to convert.

        Returns:
        - datetime_list (list of datetime): The datetime object corresponding to the MET.
        """
        return [Time.fermi_ref_time + timedelta(seconds=int(met)) for met in met_list]

    def from_met_to_datetime_str(met_list) -> list:
        """
        Convert the MET to a datetime object and return as string.

        Parameters:
        - met_list (list): The MET list to convert.

        Returns:
        - datetime_list (list of str): The datetime object corresponding to the MET, represented as strings.
        """
        return [str(Time.fermi_ref_time + timedelta(seconds=int(met))) for met in met_list]

    def remove_milliseconds_from_datetime(datetime_list) -> list:
        """
        Remove the milliseconds from the datetime object.

        Parameters:
        - datetime_list (list): The datetime list to convert.

        Returns:
        - datetime_list (list of datetime): The datetime object without milliseconds.
        """
        return [dt.replace(microsecond=0) for dt in datetime_list]

    def get_week_from_datetime(datetime_list) -> list:
        """
        Get the week number from the datetime object.

        Parameters:
        - datetime_list (list): The datetime list to convert.

        Returns:
        - week_list (list of int): The week number corresponding to the datetime.
        """
        for dt in datetime_list:
            print((datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") - Time.fermi_ref_time))
        return [(Time.fermi_ref_time + datetime(dt)).isocalendar()[1] for dt in datetime_list]

class Data():
    logger = Logger('Data').get_logger()
    """
    A class that provides utility functions for data manipulation.
    """
    
    @logger_decorator(logger)
    def get_masked_dataframe(start, stop, data, column='datetime'):
        """
        Returns the masked data within the specified time range.

        Args:
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
            data (DataFrame): The input data.
            column (str, optional): The column name representing the time. Defaults to 'datetime'.

        Returns:
            DataFrame: The masked data within the specified time range.
        """
        mask = (data[column] >= start) & (data[column] <= stop)
        masked_data = data[mask]
        return pd.DataFrame(masked_data)

    def get_excluded_dataframes(start, stop, data, column='datetime'):
        """
        Returns the excluded dataframes within the specified time range.

        Args:
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
            data (DataFrame): The input data.
            column (str, optional): The column name representing the time. Defaults to 'datetime'.

        Returns:
            list: The excluded dataframes within the specified time range.
        """
        mask = (data[column] < start) | (data[column] > stop)
        excluded_data = data[mask]
        return excluded_data

    def get_masked_data(start, stop, data, column='datetime'):
        """
        Returns the masked data within the specified time range.

        Args:
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
            data (DataFrame): The input data.
            column (str, optional): The column name representing the time. Defaults to 'datetime'.

        Returns:
            dict: The masked data within the specified time range, with column names as keys and lists of values as values.
        """
        mask = (data[column] >= start) & (data[column] <= stop)
        masked_data = data[mask]
        return {name: masked_data.field(name).tolist() for name in masked_data.names}

    @logger_decorator(logger)
    def filter_dataframe_with_run_times(initial_dataframe, run_times):
        """
        Returns the spacecraft dataframe filtered on runs times.

        Args:
            initial_dataframe (DataFrame): The initial spacecraft data.
            run_times (DataFrame): The dataframe containing run times.

        Returns:
            DataFrame: The filtered spacecraft dataframe.
        """
        df = pd.DataFrame()
        for start, end in run_times.values():
            df = pd.concat([df, Data.get_masked_dataframe(data=initial_dataframe, start=start, stop=end)], ignore_index=True)
        return df

    def convert_to_df(data_to_df) -> pd.DataFrame:
        """
        Converts the data containing the spacecraft data into a pd.DataFrame.

        Args:
            data_to_df (ndarray): The data to convert.

        Returns:
            DataFrame: The dataframe containing the spacecraft data.
        """
        return pd.DataFrame({name: data_to_df.field(name).tolist() for name in data_to_df.dtype.names})

    @logger_decorator(logger)
    def merge_dfs(first_dataframe: pd.DataFrame, second_dataframe: pd.DataFrame, on_column='datetime') -> pd.DataFrame:
        """
        Merges two dataframes based on a common column.

        Args:
            first_dataframe (DataFrame): The first dataframe.
            second_dataframe (DataFrame): The second dataframe.
            on_column (str, optional): The column to merge on. Defaults to 'datetime'.

        Returns:
            DataFrame: The merged dataframe.
        """
        return pd.merge(first_dataframe, second_dataframe, on=on_column, how='inner')

class File:
    logger = Logger('File').get_logger()

    @logger_decorator(logger)
    def write_df_on_file(df, filename=INPUTS_OUTPUTS_FILE_PATH):
        """
        Write the dataframe to a file.

        Args:
            df (DataFrame): The dataframe to write.
            filename (str, optional): The name of the file to write the dataframe to. Defaults to INPUTS_OUTPUTS_FILE_PATH.
        """
        df.to_csv(filename + '.csv', index=False)
        df.to_pickle(filename + '.pk')

    @logger_decorator(logger)
    def read_df_from_file(filename=INPUTS_OUTPUTS_FILE_PATH):
        """
        Read the dataframe from a file.

        Args:
            filename (str, optional): The name of the file to read the dataframe from. Defaults to INPUTS_OUTPUTS_FILE_PATH.

        Returns:
            DataFrame: The dataframe read from the file.
        """
        if os.path.exists(filename + '.pk'):
            return pd.read_pickle(filename + '.pk')
        else:
            return None

    @logger_decorator(logger)
    def write_on_file(data, filename):
        with open(filename, 'w') as file:
            for key, value in data.items():
                file.write(f'{key}: {value}\n')

