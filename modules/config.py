'''
Configuration file containing the folders and filenames for ACNBkg modules.
To check your configuration, run this file as a script.
'''
import os

USER = os.environ.get('USERNAME')

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# DIR = '/media/andrea/DISK4T1/ACD LAT Adelfio/'

# Folders: parent (data), children (solar, spacecraft, LAT_ACD/output runs)
DATA_FOLDER_NAME = os.path.join(DIR, 'data')

SOLAR_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'solar')

SC_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'spacecraft/LAT/weekly')

# DATA_LATACD_INPUT_FOLDER_PATH = os.path.join(DATA_FOLDER_NAME, 'LAT_ACD/Dec23-Feb24 input runs')
DATA_LATACD_INPUT_FOLDER_PATH = os.path.join(DATA_FOLDER_NAME, 'LAT_ACD/May24-Jun24 input runs')

# DATA_LATACD_FOLDER_PATH = os.path.join(DATA_FOLDER_NAME, 'LAT_ACD/Dec23-Feb24 output runs')
DATA_LATACD_FOLDER_PATH = os.path.join(DATA_FOLDER_NAME, 'LAT_ACD/May24-Jun24 output runs')

BACKGROUND_PREDICTION_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'background_prediction')

# Folders: parent (data), children (solar, spacecraft, LAT_ACD/output runs)
SOLAR_FILENAME = 'solar_activity'
SOLAR_FILE_PATH = os.path.join(SOLAR_FOLDER_NAME, SOLAR_FILENAME)

INPUTS_OUTPUTS_FILENAME = 'inputs_outputs'
INPUTS_OUTPUTS_FOLDER = os.path.join(DATA_FOLDER_NAME, 'inputs_outputs')
INPUTS_OUTPUTS_FILE_PATH = os.path.join(INPUTS_OUTPUTS_FOLDER, INPUTS_OUTPUTS_FILENAME)

LOGGING_FOLDER_NAME = 'logs'
LOGGING_FILE_NAME = f'acdbkg_{USER}.log'
LOGGING_FILE_REL_PATH = os.path.join(LOGGING_FOLDER_NAME, LOGGING_FILE_NAME)
LOGGING_FILE_PATH = os.path.join(os.getcwd(), LOGGING_FILE_REL_PATH)

MODEL_NN_SAVED_FILE_NAME = 'saved_model.keras'

if __name__ == '__main__':
    vars_copy = vars().copy()
    filtered_vars = [var for var in vars_copy if '__' not in var]
    max_len = max(map(len, filtered_vars))

    for var in filtered_vars:
        print(f"{var:<{max_len}} {vars_copy[var]}")