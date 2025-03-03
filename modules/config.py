'''
Configuration file containing the folders and filenames for ACNBkg modules.
To check your configuration, run this file as a script.
'''
import os
import pandas as pd

USER = os.environ.get('USERNAME')

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# DIR = '/media/andrea/DISK4T1/ACD LAT Adelfio/'

# Folders: parent (data), children (solar, spacecraft, LAT_ACD/output runs)
DATA_FOLDER_NAME = os.path.join(DIR, 'data')

SOLAR_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'solar')

SC_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'spacecraft/LAT/weekly')

# ACD_DATA = 'old'
# ACD_DATA = 'new'
ACD_DATA = 'newRuns_v7_extracted_v3'
DATA_LATACD_RAW_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'LAT_ACD', 'raw', ACD_DATA)
DATA_LATACD_PROCESSED_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'LAT_ACD', 'processed', ACD_DATA)

# MARK: Results
DATE_FOLDER = pd.Timestamp.date(pd.Timestamp.now()).strftime('%Y-%m-%d')
RESULTS_FOLDER_NAME = os.path.join(DIR, 'results', DATE_FOLDER)
BACKGROUND_PREDICTION_FOLDER_NAME = os.path.join(RESULTS_FOLDER_NAME, 'background_prediction')
TRIGGER_FOLDER_NAME = os.path.join(RESULTS_FOLDER_NAME, 'anomalies')
PLOT_TRIGGER_FOLDER_NAME = os.path.join(TRIGGER_FOLDER_NAME, 'plots')

# Folders: parent (data), children (solar, spacecraft, LAT_ACD/output runs)
SOLAR_FILENAME = 'solar_activity'
SOLAR_FILE_PATH = os.path.join(SOLAR_FOLDER_NAME, SOLAR_FILENAME)

INPUTS_OUTPUTS_FILENAME = 'inputs_outputs'
INPUTS_OUTPUTS_FOLDER = os.path.join(DATA_FOLDER_NAME, 'inputs_outputs')
INPUTS_OUTPUTS_FILE_PATH = os.path.join(INPUTS_OUTPUTS_FOLDER, INPUTS_OUTPUTS_FILENAME)

LOGGING_FOLDER_NAME = 'logs'
LOGGING_FOLDER_PATH = os.path.join(DIR, LOGGING_FOLDER_NAME)
LOGGING_FILE_NAME = f'{USER}.log'

FOLDERS_LIST = [DIR, 
                DATA_FOLDER_NAME, 
                SOLAR_FOLDER_NAME, 
                SC_FOLDER_NAME, 
                RESULTS_FOLDER_NAME, 
                BACKGROUND_PREDICTION_FOLDER_NAME, 
                TRIGGER_FOLDER_NAME, 
                PLOT_TRIGGER_FOLDER_NAME, 
                SOLAR_FILE_PATH, 
                INPUTS_OUTPUTS_FOLDER, 
                INPUTS_OUTPUTS_FILE_PATH, 
                LOGGING_FOLDER_PATH]
for dir in FOLDERS_LIST:
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    vars_copy = vars().copy()
    filtered_vars = [var for var in vars_copy if '__' not in var]
    max_len = max(map(len, filtered_vars))

    for var in filtered_vars:
        print(f"{var:<{max_len}} {vars_copy[var]}")