'''This is a config file to load the dataframe columns'''
y_cols_raw = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
y_cols = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
y_smooth_cols = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']
y_pred_cols = [col + '_pred' for col in y_cols_raw]

x_cols = ['SC_POSITION_0', 'SC_POSITION_1', 'SC_POSITION_2', 'LAT_GEO', 'LON_GEO',
        'RAD_GEO', 'RA_ZENITH', 'DEC_ZENITH', 'B_MCILWAIN', 'L_MCILWAIN', 
        'GEOMAG_LAT', 'LAMBDA', 'RA_SCZ', 'START', 'STOP', 'MET', 'IN_SAA',
        'LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'LIVETIME', 'DEC_SCZ', 'RA_SCX',
        'DEC_SCX', 'RA_NPOLE', 'DEC_NPOLE', 'ROCK_ANGLE',
        'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4', 'RA_SUN', 'DEC_SUN',
        'SC_VELOCITY_0', 'SC_VELOCITY_1', 'SC_VELOCITY_2',
        'LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL',
        'SOLAR', 'SUN_IS_OCCULTED', 'TIME_FROM_SAA', 'SAA_EXIT']
x_cols_excluded = y_cols + y_smooth_cols + ['datetime', 'LIVETIME', 'MET', 'START', 'STOP', 'IN_SAA']

# col_selected = inputs_outputs_df.columns