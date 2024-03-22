
from scripts.utils import File
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc
from datetime import date
import numpy as np
# Preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE, median_absolute_error as MeAE
from sklearn.preprocessing import StandardScaler
# Tensorflow, Keras
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model
# Explainability
from scripts.config import MODEL_NN_SAVED_FILE_NAME, MODEL_NN_FOLDER_NAME
from tensorflow.keras import backend as K
import tensorflow.keras.losses as losses
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from sklearn.neighbors import KNeighborsRegressor

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
    a = math.reduce_max(loss_mae_none(y_true, y_predict))  # axis=0
    return a


class NN:
    def __init__(self, df_data, col_range, col_selected):
        self.col_range = col_range
        self.col_selected = col_selected
        self.df_data = df_data
        self.y = None
        self.X = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.scaler = None
        self.nn_r = None
        self.opt_name = None
        self.units = None
        self.lr = None
    
    def create_model(self, units=200, loss_type='mean', do=0.05, opt_name='Adam', lr=0.001):
        # Load the data
        self.y = self.df_data[self.col_range].astype('float32')
        self.X = self.df_data[self.col_selected].astype('float32')
        self.units = units
        self.lr = lr
        self.do = do
        # Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=0, shuffle=True)
        # Scale
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.scaler = scaler
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        # Num of inputs as columns of table
        inputs = Input(shape=(self.X_train.shape[1],))
        hidden = Dense(units, activation='relu')(inputs)
        # hidden = BatchNormalization()(hidden)
        # hidden = Dropout(do)(hidden)

        # hidden = Dense(units, activation='relu')(hidden)
        # hidden = BatchNormalization()(hidden)
        # hidden = Dropout(do)(hidden)

        # hidden = Dense(int(units / 2), activation='relu')(hidden)
        # hidden = BatchNormalization()(hidden)
        # hidden = Dropout(do)(hidden)
        outputs = Dense(len(self.col_range), activation='relu')(hidden)

        self.nn_r = Model(inputs=[inputs], outputs=outputs)

        self.opt_name = opt_name
        # Optimizer
        if self.opt_name == 'Adam':
            opt = Adam(beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        elif self.opt_name == 'Nadam':
            opt = Nadam(beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        elif self.opt_name == 'RMSprop':
            opt = RMSprop(rho=0.6, momentum=0.0, epsilon=1e-07)
        elif self.opt_name == 'SGD':
            opt = SGD()

        if loss_type == 'max':
            # Define Loss as max_i(det_ran_error)
            loss = loss_max
        elif loss_type == 'median':
            # Define Loss as average of Median Absolute Error for each detector_range
            loss = loss_median
        elif loss_type == 'mean' or loss_type == 'mae':
            # Define Loss as average of Mean Absolute Error for each detector_range
            loss = 'mae'
            # loss = keras.losses.MeanAbsoluteError()
        elif loss_type == 'huber':
            loss = keras.losses.Huber(delta=1)
        else:
            # Define Loss as average of Mean Squared Error for each detector_range
            loss = 'mse'

        self.nn_r.compile(loss=loss, optimizer=opt)

    def scheduler(self, epoch, lr_actual):
        if epoch < 4:
            return self.lr*12.5
        if 4 <= epoch < 12:
            return self.lr*2
        if 12 <= epoch:
            return self.lr/2


    def train(self, epochs=512, bs=2000):
        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=32)
        mc = ModelCheckpoint(f'{MODEL_NN_FOLDER_NAME}/{self.units}_{self.opt_name}_{MODEL_NN_SAVED_FILE_NAME}', monitor='val_loss', mode='min',
                                verbose=0, save_best_only=True)
        
        if not self.lr:
            history = self.nn_r.fit(self.X_train, self.y_train, epochs=epochs, batch_size=bs,
                                validation_split=0.3, callbacks=[es, mc])
        else:
            call_lr = LearningRateScheduler(self.scheduler)
            history = self.nn_r.fit(self.X_train, self.y_train, epochs=epochs, batch_size=bs,
                                validation_split=0.3, callbacks=[es, mc, call_lr])
        nn_r = load_model(f'{MODEL_NN_FOLDER_NAME}/{self.units}_{self.opt_name}_{MODEL_NN_SAVED_FILE_NAME}')
        
        # Compute MAE per each detector and range
        pred_train = nn_r.predict(self.X_train)
        pred_test = nn_r.predict(self.X_test)
        idx = 0
        text_mae = ""
        for col in self.col_range:
            mae_tr = MAE(self.y_train.iloc[:, idx], pred_train[:, idx])
            mae_te = MAE(self.y_test.iloc[:, idx], pred_test[:, idx])
            diff_i = (self.y_test.iloc[:, idx] - pred_test[:, idx]).mean()
            meae_tr = MeAE(self.y_train.iloc[:, idx], pred_train[:, idx])
            meae_te = MeAE(self.y_test.iloc[:, idx], pred_test[:, idx])
            diff_i_m = (self.y_test.iloc[:, idx] - pred_test[:, idx]).median()
            text_tr = "MAE train of " + col + " : %0.3f" % (mae_tr)
            text_te = "MAE test of " + col + " : %0.3f" % (mae_te)
            test_diff_i = "diff test - pred " + col + " : %0.3f" % (diff_i)
            text_tr_m = "MeAE train of " + col + " : %0.3f" % (meae_tr)
            text_te_m = "MeAE test of " + col + " : %0.3f" % (meae_te)
            test_diff_i_m = "med diff test - pred " + col + " : %0.3f" % (diff_i_m)
            text_mae += text_tr + '    ' + text_te + '    ' + test_diff_i + '    ' + \
                        text_tr_m + '    ' + text_te_m + '    ' + test_diff_i_m + '\n'
            idx = idx + 1

        # plot training history
        plt.plot(history.history['loss'][4:], label=f'train {self.units} {self.opt_name}')
        plt.plot(history.history['val_loss'][4:], label=f'test {self.units} {self.opt_name}')
        plt.legend()

        name_model = f'model_{self.opt_name}_do_{self.do}_{round(nn_r.evaluate(self.X_test, self.y_test), 2)}_units_{self.units}_{date.today()}'
        nn_r.save(MODEL_NN_FOLDER_NAME + name_model + '.keras')
        self.nn_r = nn_r
        # Save figure of performance
        plt.savefig(MODEL_NN_FOLDER_NAME + name_model + '.png')
        # open text file and write mae performance
        text_file = open(MODEL_NN_FOLDER_NAME + name_model + '.txt', "w")
        text_file.write(text_mae)
        text_file.close()

    def predict(self, start = 0, end = -1):
        df_data = self.df_data[start:end]
        pred_x_tot = self.nn_r.predict(self.scaler.transform(df_data[self.col_selected]))
        ts = df_data['datetime']
        gc.collect()

        df_ori = df_data[self.col_range].reset_index(drop=True)
        y_pred = pd.DataFrame(pred_x_tot, columns=self.col_range)
        df_ori['timestamp'] = ts.values
        y_pred['timestamp'] = ts.values
        df_ori['met'] = df_data['MET'].values
        y_pred['met'] = df_data['MET'].values

        df_ori.reset_index(drop=True, inplace=True)
        y_pred.reset_index(drop=True, inplace=True)

        File.write_df_on_file(df_ori, MODEL_NN_FOLDER_NAME + '/frg')
        File.write_df_on_file(y_pred, MODEL_NN_FOLDER_NAME + '/bkg')

    def plot(self, df_ori, y_pred, det_rng='top'):
        with sns.plotting_context("talk"):
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 12))  # , tight_layout=True)
            fig.subplots_adjust(hspace=0)
            fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori['timestamp']).iloc[0]))

            axs[0].plot(pd.to_datetime(df_ori['timestamp']), df_ori[det_rng], 'k-.')
            axs[0].plot(pd.to_datetime(df_ori['timestamp']), y_pred[det_rng], 'r-')

            axs[0].set_title('foreground and background')
            axs[0].set_ylabel('Count Rate')

            axs[1].plot(pd.to_datetime(df_ori['timestamp']),
                        df_ori[det_rng] - y_pred[det_rng], 'k-.')
            axs[1].plot(pd.to_datetime(df_ori['timestamp']).ffill(),
                        df_ori['met'].ffill() * 0, 'k-')
            axs[1].set_xlabel('time (YYYY-MM-DD hh:mm:ss)')
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

# from sklearn.neighbors import KNeighborsRegressor, check_array, _get_weights

# class MedianKNNRegressor(KNeighborsRegressor):
#     def predict(self, X):
#         X = check_array(X, accept_sparse='csr')

#         neigh_dist, neigh_ind = self.kneighbors(X)

#         weights = _get_weights(neigh_dist, self.weights)

#         _y = self._y
#         if _y.ndim == 1:
#             _y = _y.reshape((-1, 1))

#         ######## Begin modification
#         if weights is None:
#             y_pred = np.median(_y[neigh_ind], axis=1)
#         else:
#             # y_pred = weighted_median(_y[neigh_ind], weights, axis=1)
#             raise NotImplementedError("weighted median")
#         ######### End modification

#         if self._y.ndim == 1:
#             y_pred = y_pred.ravel()

#         return y_pred    

class MedianKNeighborsRegressor(KNeighborsRegressor):
    def predict(self, X):
        """Predict the target for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
            Target values.
        """
        if self.weights == "uniform":
            # In that case, we do not need the distances to perform
            # the weighting so we do not compute them.
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        weights = None

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.median(_y[neigh_ind], axis=1)

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred