'''
Configuration file for ACNBkg.scripts
'''
import os

USER = os.environ.get('USERNAME')
DIR = os.path.dirname(os.path.realpath(__file__))

DATA_REL_FOLDER_PATH = '/data/nfs/farm/g/glast/g/CRE/test_sm/BATgrbs'
DATA_FOLDER_PATH = os.path.join(os.getcwd(), DIR, DATA_REL_FOLDER_PATH)

LOGGING_FILE_NAME = 'acnbkg.log'
LOGGING_FOLDER_NAME = 'logs'
LOGGING_FILE_REL_PATH = os.path.join(DIR, LOGGING_FOLDER_NAME, LOGGING_FILE_NAME)
LOGGING_FILE_PATH = os.path.join(os.getcwd(), LOGGING_FILE_REL_PATH)

SC_FILENAME = 'lat_spacecraft_merged.fits'
SC_FOLDER_NAME = 'spacecraft'
SC_FILE_REL_PATH = os.path.join(DIR, SC_FOLDER_NAME, SC_FILENAME)
SC_FILE_PATH = os.path.join(os.getcwd(), SC_FILE_REL_PATH)
