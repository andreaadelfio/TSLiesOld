"""Utils module for the ACNBkg project."""
from datetime import datetime, timedelta

fermi_ref_time = datetime(2001, 1, 1, 0, 0, 0)


def from_met_to_datetime(met_list) -> list:
    """
    Convert the MET to a datetime object.
    
    Parameters:
    - met_list (list): The MET list to convert.
    
    Returns:
    - datetime_list (list of datetime): The datetime object corresponding to the MET.
    """
    return [fermi_ref_time + timedelta(seconds=int(met)) for met in met_list]


def from_met_to_datetime_str(met_list) -> list:
    """
    Convert the MET to a datetime object and return as string.
    
    Parameters:
    - met_list (list): The MET list to convert.
    
    Returns:
    - datetime_list (list of str): The datetime object corresponding to the MET, represented as strings.
    """
    return [str(fermi_ref_time + timedelta(seconds=int(met))) for met in met_list]
