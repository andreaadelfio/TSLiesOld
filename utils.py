"""Utils module for the ACNBkg project."""
from datetime import datetime, timedelta
import pandas as pd
from config import INPUTS_OUTPUTS_FILE_PATH

class Time:
    fermi_ref_time = datetime(2001, 1, 1, 0, 0, 0)


    def from_met_to_datetime(met_list) -> list:
        """
        Convert the MET to a datetime object.
        
        Parameters:
        - met_list (list): The MET list to convert.
        
        Returns:
        - datetime_list (list of datetime): The datetime object corresponding to the MET.
        """
        fermi_ref_time = datetime(2001, 1, 1, 0, 0, 0)
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

class Data():
    def get_masked_dataframe(start, stop, data, column = 'datetime'):
        """
        Returns the masked data within the specified time range.
        
        Args:
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
        
        Returns:
            numpy.ndarray: The masked data within the specified time range.
        """
        mask = (data[column] >= start) & (data[column] <= stop)
        masked_data = data[mask]
        return pd.DataFrame(masked_data)

    def get_excluded_dataframes(start, stop, data, column = 'datetime'):
        """
        Returns the excluded dataframes within the specified time range.
        
        Args:
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
        
        Returns:
            list: The excluded dataframes within the specified time range.
        """
        mask = (data[column] < start) | (data[column] > stop)
        excluded_data = data[mask]
        return excluded_data

    def get_masked_data(start, stop, data, column = 'datetime'):
        """
        Returns the masked data within the specified time range.
        
        Args:
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
        
        Returns:
            numpy.ndarray: The masked data within the specified time range.
        """
        mask = (data[column] >= start) & (data[column] <= stop)
        masked_data = data[mask]
        return {name: masked_data.field(name).tolist() for name in masked_data.names}

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
        Converts the data containing the spacecraft data in a pd.DataFrame.
        
        Returns:
            pandas.DataFrame: The dataframe containing the spacecraft data.
        """
        return pd.DataFrame({name: data_to_df.field(name).tolist() for name in data_to_df.dtype.names})
    
    def merge_dfs(first_dataframe: pd.DataFrame, second_dataframe: pd.DataFrame, on_column = 'datetime') -> pd.DataFrame:
        return pd.merge(first_dataframe, second_dataframe, on = on_column, how = 'outer')
    
    def write_df_on_file(df, filename = INPUTS_OUTPUTS_FILE_PATH):
        """
        Write the dataframe to a file.
        
        Args:
            df (DataFrame): The dataframe to write.
            filename (str): The name of the file to write the dataframe to.
        """
        df.to_csv(filename, index=False)

    def read_df_from_file(filename = INPUTS_OUTPUTS_FILE_PATH):
        """
        Read the dataframe from a file.
        
        Args:
            filename (str): The name of the file to read the dataframe from.
        
        Returns:
            DataFrame: The dataframe read from the file.
        """
        return pd.read_csv(filename)