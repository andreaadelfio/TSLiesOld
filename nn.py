
from utils import File
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import utils
# Standard packages
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
import pandas as pd
import gc
from astropy.time import Time
from datetime import date
# Preprocess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE, median_absolute_error as MeAE
from sklearn.preprocessing import StandardScaler
# Tensorflow, Keras
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
# Explainability
import shap
from config import MODEL_NN_SAVED_FILE_PATH, MODEL_NN_FOLDER_NAME
from tensorflow.keras import backend as K
import tensorflow.keras.losses as losses
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


def loss_median(y_t, f):
    """
    Median Absolute Error. If q=0.5 the metric is Median Absolute Error.
    :param y_t: target value
    :param f: predicted value
    :return: Median Absolute Error
    """
    q = 0.50
    y_pred = ops.convert_to_tensor_v2_with_dispatch(f)
    y_true = math_ops.cast(y_t, y_pred.dtype)
    err = (y_true - y_pred)
    # err = (math_ops.pow(y_true, 1.5) - math_ops.pow(y_pred, 1.5))
    # return K.mean(K.maximum(q * err, (q - 1) * err), axis=-1)
    return K.mean(math_ops.maximum(q * err, (q - 1) * err), axis=-1) # + \
          # math_ops.sqrt(K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1))


def loss_max(y_true, y_predict):
    """
    Take the maximum of the MAE detectors.
    :param y_true: y target
    :param y_predict: y predicted by the NN
    :return: max_i(MAE_i)
    """
    # Define Loss as max_i(det_ran_error)
    loss_mae_none = losses.MeanAbsoluteError(reduction=losses.Reduction.NONE)
    a = tf.math.reduce_max(loss_mae_none(y_true, y_predict))  # axis=0
    return a


class NN:
    def __init__(self, df_data):
        self.col_range = ['top','Xpos','Xneg','Ypos','Yneg']
        self.col_sat_pos = ['SC_POSITION_0','SC_POSITION_1','SC_POSITION_2','LAT_GEO','LON_GEO','RAD_GEO','RA_ZENITH','DEC_ZENITH','B_MCILWAIN','L_MCILWAIN','GEOMAG_LAT','LAMBDA','IN_SAA','RA_SCZ','DEC_SCZ','RA_SCX','DEC_SCX','RA_NPOLE','DEC_NPOLE','ROCK_ANGLE','LAT_MODE','LAT_CONFIG','DATA_QUAL','LIVETIME','QSJ_1','QSJ_2','QSJ_3','QSJ_4','RA_SUN','DEC_SUN','SC_VELOCITY_0','SC_VELOCITY_1','SC_VELOCITY_2','SOLAR']
        # self.col_sat_pos = ['SC_POSITION_0','SC_POSITION_1','SC_POSITION_2','LAT_GEO','LON_GEO','RAD_GEO','RA_ZENITH','DEC_ZENITH','B_MCILWAIN','L_MCILWAIN','GEOMAG_LAT','LAMBDA','IN_SAA','RA_SCZ','DEC_SCZ','RA_SCX','DEC_SCX','RA_NPOLE','DEC_NPOLE','ROCK_ANGLE','LAT_MODE','LAT_CONFIG','DATA_QUAL','LIVETIME','QSJ_1','QSJ_2','QSJ_3','QSJ_4','RA_SUN','DEC_SUN','SC_VELOCITY_0','SC_VELOCITY_1','SC_VELOCITY_2']

        self.col_selected = self.col_sat_pos
        self.df_data = df_data
    
    def train(self, loss_type='mean', units=200, epochs=512, lr=0.001, bs=2000, do=0.05):
        # Load the data
        y = self.df_data[self.col_range].astype('float32')
        X = self.df_data[self.col_selected].astype('float32')
        # Splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)
        # Scale
        scaler = StandardScaler()
        scaler.fit(X_train)
        self.scaler = scaler
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # Num of inputs as columns of table
        inputs = tf.keras.Input(shape=(X_train.shape[1],))

        hidden = Dense(units, activation='relu')(inputs)
        # hidden = tf.keras.layers.BatchNormalization()(hidden)
        # hidden = Dropout(do)(hidden)

        # hidden = Dense(units, activation='relu')(hidden)
        # hidden = tf.keras.layers.BatchNormalization()(hidden)
        # hidden = Dropout(do)(hidden)

        # hidden = Dense(int(units / 2), activation='relu')(hidden)
        # hidden = tf.keras.layers.BatchNormalization()(hidden)
        # hidden = Dropout(do)(hidden)
        outputs = Dense(len(self.col_range), activation='relu')(hidden)

        nn_r = tf.keras.Model(inputs=[inputs], outputs=outputs)
        # Optimizer
        opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        # opt = tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        # opt = tf.keras.optimizers.RMSprop( learning_rate=0.002, rho=0.6, momentum=0.0, epsilon=1e-07)

        if loss_type == 'max':
            # Define Loss as max_i(det_ran_error)
            loss = loss_max
        elif loss_type == 'median':
            # Define Loss as average of Median Absolute Error for each detector_range
            loss = loss_median
        elif loss_type == 'mean' or loss_type == 'mae':
            # Define Loss as average of Mean Absolute Error for each detector_range
            loss = 'mae'
            # loss = tf.keras.losses.MeanAbsoluteError()
        elif loss_type == 'huber':
            loss = tf.keras.losses.Huber(delta=1)
        else:
            # Define Loss as average of Mean Squared Error for each detector_range
            loss = 'mse'

        nn_r.compile(loss=loss, optimizer=opt)

        def scheduler(epoch, lr_actual):
            if epoch < 4:
                return lr*12.5
            if 4 <= epoch < 12:
                return lr*2
            if 12 <= epoch:
                return lr/2

        call_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

        # Fitting the model
        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=32)
        mc = ModelCheckpoint(MODEL_NN_SAVED_FILE_PATH, monitor='val_loss', mode='min',
                                verbose=0, save_best_only=True)
        
        history = nn_r.fit(X_train, y_train, epochs=epochs, batch_size=bs,
                            validation_split=0.3, callbacks=[es, mc])#, call_lr])
        nn_r = load_model(MODEL_NN_SAVED_FILE_PATH)
        
        # Insert loss result in model name
        loss_test = round(nn_r.evaluate(X_test, y_test), 2)
        # TODO set a proper name of the model
        today = date.today()
        name_model = 'model_' + str(loss_test) + '_' + str(today)

        # Predict the model
        pred_train = nn_r.predict(X_train)
        pred_test = nn_r.predict(X_test)

        # Compute MAE per each detector and range
        idx = 0
        text_mae = ""
        for i in self.col_range:
            mae_tr = MAE(y_train.iloc[:, idx], pred_train[:, idx])
            mae_te = MAE(y_test.iloc[:, idx], pred_test[:, idx])
            diff_i = (y_test.iloc[:, idx] - pred_test[:, idx]).mean()
            meae_tr = MeAE(y_train.iloc[:, idx], pred_train[:, idx])
            meae_te = MeAE(y_test.iloc[:, idx], pred_test[:, idx])
            diff_i_m = (y_test.iloc[:, idx] - pred_test[:, idx]).median()
            text_tr = "MAE train of " + i + " : %0.3f" % (mae_tr)
            text_te = "MAE test of " + i + " : %0.3f" % (mae_te)
            test_diff_i = "diff test - pred " + i + " : %0.3f" % (diff_i)
            text_tr_m = "MeAE train of " + i + " : %0.3f" % (meae_tr)
            text_te_m = "MeAE test of " + i + " : %0.3f" % (meae_te)
            test_diff_i_m = "med diff test - pred " + i + " : %0.3f" % (diff_i_m)
            text_mae += text_tr + '    ' + text_te + '    ' + test_diff_i + '    ' + \
                        text_tr_m + '    ' + text_te_m + '    ' + test_diff_i_m + '\n'
            idx = idx + 1

        # plot training history
        plt.plot(history.history['loss'][4:], label='train')
        plt.plot(history.history['val_loss'][4:], label='test')
        plt.legend()

        nn_r.save(MODEL_NN_FOLDER_NAME + name_model + '.keras')
        self.nn_r = nn_r
        # Save figure of performance
        plt.savefig(MODEL_NN_FOLDER_NAME + name_model + '.png')
        # open text file and write mae performance
        text_file = open(MODEL_NN_FOLDER_NAME + name_model + '.txt', "w")
        text_file.write(text_mae)
        text_file.close()

    def predict(self):
        pred_x_tot = self.nn_r.predict(self.scaler.transform(self.df_data[self.col_selected]))
        ts = self.df_data['datetime']
        gc.collect()

        # # # Generate a dataset for trigger algorithm
        # Original bkg + ssa + met
        df_ori = self.df_data[self.col_range].reset_index(drop=True)
        # Prediction of the bkg
        y_pred = pd.DataFrame(pred_x_tot, columns=self.col_range)
        df_ori['timestamp'] = ts
        y_pred['timestamp'] = ts
        df_ori['met'] = self.df_data['MET'].values
        y_pred['met'] = self.df_data['MET'].values


        df_ori.reset_index(drop=True, inplace=True)
        y_pred.reset_index(drop=True, inplace=True)

        # Save the data
        File.write_df_on_file(df_ori, MODEL_NN_FOLDER_NAME + '/frg')
        File.write_df_on_file(y_pred, MODEL_NN_FOLDER_NAME + '/bkg')

    def plot(self, df_ori, y_pred, det_rng='top'):
        # Plot a particular zone and det_rng
        # df_ori = pd.read_csv(MODEL_NN_FOLDER_NAME + '/frg' + '.csv')
        # y_pred = pd.read_csv(MODEL_NN_FOLDER_NAME + '/bkg' + '.csv')
        # Plot frg, bkg and residual for det_rng
        with sns.plotting_context("talk"):
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 12))  # , tight_layout=True)
            # Remove horizontal space between axes
            fig.subplots_adjust(hspace=0)
            fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori['timestamp']).iloc[0]))

            # Plot each graph, and manually set the y tick values
            axs[0].plot(pd.to_datetime(df_ori['timestamp']), df_ori[det_rng], 'k-.')
            axs[0].plot(pd.to_datetime(df_ori['timestamp']), y_pred[det_rng], 'r-')

            axs[0].set_title('foreground and background')
            #axs[0].set_xlabel('time')
            axs[0].set_ylabel('Count Rate')

            axs[1].plot(pd.to_datetime(df_ori['timestamp']),
                        df_ori[det_rng] - y_pred[det_rng], 'k-.')
            axs[1].plot(pd.to_datetime(df_ori['timestamp']).ffill(),
                        df_ori['met'].ffill() * 0, 'k-')
            axs[1].set_xlabel('time (YYYY-MM-DD hh:mm:ss)')
            # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            # axs[1].xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=0)
            axs[1].set_ylabel('Residuals')

        # Plot y_pred vs y_true
        with sns.plotting_context("talk"):
            fig = plt.figure()
            fig.set_size_inches(24, 12)
            plt.axis('equal')
            plt.plot(df_ori[self.col_range], y_pred[self.col_range], '.', alpha=0.2)
            min_y, max_y = min(y_pred[self.col_range].min()), max(y_pred[self.col_range].max())
            plt.plot([min_y, max_y], [min_y, max_y], '-')
            plt.xlabel('True signal')
            plt.ylabel('Predicted signal')
        plt.legend(self.col_range)
