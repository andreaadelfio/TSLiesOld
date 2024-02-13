import pandas as pd
from spacecraftopener import SpacecraftOpener
from catalogreader import CatalogReader
from plotter import Plotter


cr = CatalogReader(start = 0, end = 20)
grbs_dirs = cr.get_grbs_dirs()
grb_dict = cr.get_grb_dict(grbs_dirs, smooth = True)
for key, signals in grb_dict.items():
    label = f'{key}, met {cr.get_grb_times()[key][0]}'
    # Plotter(xy = signals, label = label).plot_tiles(lw = 0.5, with_smooth = True)

sco = SpacecraftOpener()
sco.open()
data = sco.get_data()
for met, start, end in cr.get_grb_times().values():
    print(met, met - start, met + end)
    # masked_df = pd.concat([masked_df, sco.get_masked_dataframe(met - start, met + end)], ignore_index=True)
    data = sco.get_excluded_dataframes(data, met - start, met + end)

data = sco.get_masked_dataframe(data = data, start = data['START'][0], stop = met + 50000)
Y_KEY = 'IN_SAA'
Plotter(x = data['START'], y = data[Y_KEY], label = f'spacecraft {Y_KEY}').plot(marker = ',')
