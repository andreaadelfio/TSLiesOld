'''
Configuration file containing the folders and filenames for ACNBkg modules.
To check your configuration, run this file as a script.
'''
import os
import pandas as pd

USER = os.environ.get('USERNAME')

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# DIR = '/media/andrea/DISK4T1/ACD LAT Adelfio/'

# MARK: Data Folders: parent (data), children (solar, spacecraft, LAT_ACD)
DATA_FOLDER_NAME = os.path.join(DIR, 'data')

LATACD_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'LAT_ACD')

SOLAR_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'solar')

SC_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'spacecraft/LAT/weekly')

# ACD_DATA = 'old'
# ACD_DATA = 'new'
# ACD_DATA = 'new_old'
# ACD_DATA = 'maldera'
# ACD_DATA = 'newRuns_v16'
# ACD_DATA = 'for_alberto'
# ACD_DATA = 'newRuns_v7_extracted_v3'
# ACD_DATA = 'test'
ACD_DATA = 'new_with_correct_triggs'
DATA_LATACD_RAW_FOLDER_NAME = os.path.join(LATACD_FOLDER_NAME, 'raw', ACD_DATA)
DATA_LATACD_PROCESSED_FOLDER_NAME = os.path.join(LATACD_FOLDER_NAME, 'processed', ACD_DATA)
INPUTS_OUTPUTS_FILENAME = 'inputs_outputs'
INPUTS_OUTPUTS_FILE_PATH = os.path.join(DATA_LATACD_PROCESSED_FOLDER_NAME, INPUTS_OUTPUTS_FILENAME)

# MARK: Logging
LOGGING_FOLDER_NAME = 'logs'
LOGGING_FOLDER_PATH = os.path.join(DIR, LOGGING_FOLDER_NAME)
LOGGING_FILE_NAME = f'{USER}.log'

# MARK: Results
now = pd.Timestamp.now()
DATE_FOLDER = pd.Timestamp.date(now).strftime('%Y-%m-%d')
TIME_FOLDER = pd.Timestamp.time(now).strftime('%H%M')
RESULTS_FOLDER_NAME = os.path.join(DIR, 'results', DATE_FOLDER)
BACKGROUND_PREDICTION_FOLDER_NAME = os.path.join(RESULTS_FOLDER_NAME, 'background_prediction')
TRIGGER_FOLDER_NAME = os.path.join(RESULTS_FOLDER_NAME, 'anomalies')
PLOT_TRIGGER_FOLDER_NAME = os.path.join(TRIGGER_FOLDER_NAME, TIME_FOLDER, 'plots')

FOLDERS_LIST = [DIR,
                DATA_FOLDER_NAME,
                SOLAR_FOLDER_NAME,
                SC_FOLDER_NAME,
                RESULTS_FOLDER_NAME,
                BACKGROUND_PREDICTION_FOLDER_NAME,
                TRIGGER_FOLDER_NAME,
                PLOT_TRIGGER_FOLDER_NAME,
                INPUTS_OUTPUTS_FILE_PATH,
                LOGGING_FOLDER_PATH]

if __name__ == '__main__':
    vars_copy = vars().copy()
    filtered_vars = [var for var in vars_copy if '__' not in var and isinstance(vars_copy[var], str)]
    max_len = max(map(len, filtered_vars))

    for var in filtered_vars:
        print(f"{var:>{max_len}} : {vars_copy[var]}")