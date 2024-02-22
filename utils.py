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
    datetime_list = []
    for met in met_list:
        datetime_list.append(fermi_ref_time + timedelta(seconds=int(met)))
    return datetime_list


def from_met_to_datetime_str(met_list) -> list:
    """
    Convert the MET to a datetime object.
    
    Parameters:
    - met_list (list): The MET list to convert.
    
    Returns:
    - datetime_list (list of datetime): The datetime object corresponding to the MET.
    """
    datetime_list = []
    for met in met_list:
        datetime_list.append(str(fermi_ref_time + timedelta(seconds=int(met))))
    return datetime_list

