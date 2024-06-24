'''
Configuration file for ACNBkg modules
'''
import os
import pathlib

USER = os.environ.get('USERNAME')

DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent
# DIR = '/media/andrea/DISK4T1/ACD LAT Adelfio/'

# Folders: parent (data), children (solar, spacecraft, LAT_ACD/output runs)
DATA_FOLDER_NAME = os.path.join(DIR, 'data')
SOLAR_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'solar')
SC_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'spacecraft/LAT/weekly')
DATA_LATACD_FOLDER_PATH = os.path.join(DATA_FOLDER_NAME, 'LAT_ACD/output runs')
MODEL_NN_FOLDER_NAME = os.path.join(DATA_FOLDER_NAME, 'model_nn')

SOLAR_FILENAME = 'solar_activity'
SOLAR_FILE_PATH = os.path.join(SOLAR_FOLDER_NAME, SOLAR_FILENAME)

TILE_SIGNAL_FILENAME = 'tile_signal'
TILE_SIGNAL_FILE_PATH = os.path.join(DATA_LATACD_FOLDER_PATH, TILE_SIGNAL_FILENAME)

SC_FILENAME = 'sc'
SC_FILE_PATH = os.path.join(SC_FOLDER_NAME, SC_FILENAME)

INPUTS_OUTPUTS_FILENAME = 'inputs_outputs'
INPUTS_OUTPUTS_FOLDER = os.path.join(DATA_FOLDER_NAME, 'inputs_outputs')
INPUTS_OUTPUTS_FILE_PATH = os.path.join(INPUTS_OUTPUTS_FOLDER, INPUTS_OUTPUTS_FILENAME)
INPUTS_OUTPUTS_PK_FOLDER = os.path.join(DATA_FOLDER_NAME, 'inputs_outputs', 'pk')


LOGGING_FOLDER_NAME = 'logs'
LOGGING_FILE_NAME = 'acdbkg.log'
LOGGING_FILE_REL_PATH = os.path.join(LOGGING_FOLDER_NAME, LOGGING_FILE_NAME)
LOGGING_FILE_PATH = os.path.join(os.getcwd(), LOGGING_FILE_REL_PATH)

MODEL_NN_SAVED_FILE_NAME = 'saved_model.keras'
