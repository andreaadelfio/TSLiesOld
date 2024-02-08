# from astropy.io import fits
import pandas as pd
# import numpy as np
from spacecraftopener import SpacecraftOpener


sco = SpacecraftOpener()
sco.open()

masked_df1 = sco.get_masked_dataframe(239557417, 239557500)
masked_df2 = sco.get_masked_dataframe(239557500, 239558800)
masked_df = pd.concat([masked_df1, masked_df2])
print(sco.get_data_columns()) # https://fermi.gsfc.nasa.gov/ssc/data/p7rep/analysis/documentation/Cicerone/Cicerone_Data/LAT_Data_Columns.html#SpacecraftFile
print(masked_df[['START', 'STOP', 'SC_POSITION']])
