
import os
from scripts.utils import File
import matplotlib.pyplot as plt
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
from scripts.config import MODEL_NN_FOLDER_NAME
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import tensorflow.keras.losses as losses
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor


# def loss_median(y_t, f):
#     """
#     Median Absolute Error. If q=0.5 the metric is Median Absolute Error.
#     :param y_t: target value
#     :param f: predicted value
#     :return: Median Absolute Error
#     """
#     q = 0.50
#     y_pred = ops.convert_to_tensor_v2_with_dispatch(f)
#     y_true = math_ops.cast(y_t, y_pred.dtype)
#     err = (y_true - y_pred)
#     # err = (math_ops.pow(y_true, 1.5) - math_ops.pow(y_pred, 1.5))
#     # return K.mean(K.maximum(q * err, (q - 1) * err), axis=-1)
#     return K.mean(math_ops.maximum(q * err, (q - 1) * err), axis=-1) # + \
#           # math_ops.sqrt(K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1))


# def loss_max(y_true, y_predict):
#     """
#     Take the maximum of the MAE detectors.
#     :param y_true: y target
#     :param y_predict: y predicted by the NN
#     :return: max_i(MAE_i)
#     """
#     # Define Loss as max_i(det_ran_error)
#     loss_mae_none = losses.MeanAbsoluteError(reduction=losses.Reduction.NONE)
#     a = math.reduce_max(loss_mae_none(y_true, y_predict))  # axis=0
#     return a


class NN:
    def __init__(self, df_data, col_range, col_selected):
        self.col_range = col_range
        self.col_selected = col_selected
        self.df_data = df_data

        if not os.path.exists(f'{MODEL_NN_FOLDER_NAME}'):
            os.makedirs(f'{MODEL_NN_FOLDER_NAME}')
        self.y = None
        self.X = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.scaler = None
        self.nn_r = None
        self.text = None
        self.model_id = None

        self.params = None
        self.norm = None
        self.drop = None
        self.units_1 = None
        self.units_2 = None
        self.units_3 = None
        self.bs = None
        self.do = None
        self.opt_name = None
        self.lr = None
        self.loss_type = None
        self.epochs = None
        self.mae_tr_list = None

    def trim_hyperparams_combinations(self, hyperparams_combinations):
        hyperparams_combinations_tmp = []
        seen = set()
        if os.path.exists(MODEL_NN_FOLDER_NAME + '/models_params.csv'):
            os.remove(MODEL_NN_FOLDER_NAME + '/models_params.csv')
        with open(MODEL_NN_FOLDER_NAME + '/models_params.csv', 'a') as f:
            f.write('\t'.join(['model_id', 'units_1', 'units_2', 'units_3', 'norm', 'drop', 'epochs', 'bs', 'do', 'opt_name', 'lr', 'loss_type', 'top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']) + '\n')
        for model_id, (units_1, units_2, units_3, norm, drop, epochs, bs, do, opt_name, lr, loss_type) in enumerate(hyperparams_combinations):
            # sorted_tuple = tuple(sorted([units_1, units_2, units_3]))
            # if sorted_tuple not in seen and sorted_tuple[0:-1] == (0, 0) and sorted_tuple[-1] != 0:
            #     seen.add(sorted_tuple)
            # else:
            #     model_id -= 1
            #     continue
            if units_1 == 0 and units_2 == 0 and units_3 == 0:
                model_id -= 1
                continue
            hyperparams_combinations_tmp.append((model_id, units_1, units_2, units_3, norm, drop, epochs, bs, do, opt_name, lr, loss_type))
        return hyperparams_combinations_tmp

    def set_hyperparams(self, params):
        self.params = params
        self.model_id = params['model_id']
        if not os.path.exists(f'{MODEL_NN_FOLDER_NAME}/{self.model_id}'):
            os.makedirs(f'{MODEL_NN_FOLDER_NAME}/{self.model_id}')
        self.units_1 = params['units_1']
        self.units_2 = params['units_2']
        self.units_3 = params['units_3']
        self.bs = params['bs']
        self.do = params['do']
        self.opt_name = params['opt_name']
        self.lr = params['lr']
        self.loss_type = params['loss_type']
        self.epochs = params['epochs']
        self.norm = params['norm']
        self.drop = params['drop']
        self.mae_tr_list = []

    def create_model(self):
        # Load the data
        self.y = self.df_data[self.col_range].astype('float32')
        self.X = self.df_data[self.col_selected].astype('float32')
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
        if self.units_1:
            hidden = Dense(self.units_1, activation='relu')(inputs)
            if self.norm:
                hidden = BatchNormalization()(hidden)
            if self.drop:
                hidden = Dropout(self.do)(hidden)
        else:
            hidden = inputs


        if self.units_2:
            hidden = Dense(self.units_2, activation='relu')(hidden)
            if self.norm:
                hidden = BatchNormalization()(hidden)
            if self.drop:
                hidden = Dropout(self.do)(hidden)

        if self.units_3:
            hidden = Dense(self.units_3, activation='relu')(hidden)
            if self.norm:
                hidden = BatchNormalization()(hidden)
            if self.drop:
                hidden = Dropout(self.do)(hidden)
        outputs = Dense(len(self.col_range), activation='linear')(hidden)

        self.nn_r = Model(inputs=[inputs], outputs=outputs)
        plot_model(self.nn_r, to_file=f'{MODEL_NN_FOLDER_NAME}/{self.model_id}/schema.png', show_shapes=True, show_layer_names=True, rankdir='LR')
        # Optimizer
        if self.opt_name == 'Adam':
            opt = Adam(beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        elif self.opt_name == 'Nadam':
            opt = Nadam(beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        elif self.opt_name == 'RMSprop':
            opt = RMSprop(rho=0.6, momentum=0.0, epsilon=1e-07)
        elif self.opt_name == 'SGD':
            opt = SGD()

        if self.loss_type == 'max':
            # Define Loss as max_i(det_ran_error)
            loss = loss_max
        elif self.loss_type == 'median':
            # Define Loss as average of Median Absolute Error for each detector_range
            loss = loss_median
        elif self.loss_type == 'mean' or self.loss_type == 'mae':
            # Define Loss as average of Mean Absolute Error for each detector_range
            loss = 'mae'
            # loss = keras.losses.MeanAbsoluteError()
        elif self.loss_type == 'huber':
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


    def train(self):
        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=10, start_from_epoch=50)
        mc = ModelCheckpoint(f'{MODEL_NN_FOLDER_NAME}/{self.model_id}/model.keras', monitor='val_loss', mode='min',
                                verbose=0, save_best_only=True)

        if not self.lr:
            history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs,
                                validation_split=0.3, callbacks=[es, mc])
        else:
            call_lr = LearningRateScheduler(self.scheduler)
            history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs,
                                validation_split=0.3, callbacks=[es, mc, call_lr])
        nn_r = load_model(f'{MODEL_NN_FOLDER_NAME}/{self.model_id}/model.keras')

        # Compute MAE per each detector and range
        pred_train = nn_r.predict(self.X_train)
        pred_test = nn_r.predict(self.X_test)
        idx = 0
        text = ''
        for col in self.col_range:
            mae_tr = MAE(self.y_train.iloc[:, idx], pred_train[:, idx])
            self.mae_tr_list.append(mae_tr)
            mae_te = MAE(self.y_test.iloc[:, idx], pred_test[:, idx])
            diff_i = (self.y_test.iloc[:, idx] - pred_test[:, idx])
            mean_diff_i = (diff_i).mean()
            meae_tr = MeAE(self.y_train.iloc[:, idx], pred_train[:, idx])
            meae_te = MeAE(self.y_test.iloc[:, idx], pred_test[:, idx])
            median_diff_i = (diff_i).median()
            text += f"MAE_train_{col} : {mae_tr:0.5f}\t" + \
                    f"MAE_test_{col} : {mae_te:0.5f}\t" + \
                    f"mean_diff_test_pred_{col} : {mean_diff_i:0.5f}\t" + \
                    f"MeAE_train_{col} {meae_tr:0.5f}\t" + \
                    f"MeAE_test_{col} {meae_te:0.5f}\t" + \
                    f"median_diff_test_pred_{col} {median_diff_i:0.5f}\n"
            idx = idx + 1

        # plot training history
        plt.figure("history_loss", layout="tight")
        plt.plot(history.history['loss'][4:], label=f'train {self.units_1}-{self.units_2}-{self.units_3} {self.opt_name}')
        plt.plot(history.history['val_loss'][4:], label=f'test {self.units_1}-{self.units_2}-{self.units_3} {self.opt_name}')
        plt.legend()
        plt.figure("history_accuracy", layout="tight")
        plt.plot(history.history['accuracy'][4:], label=f'train {self.units_1}-{self.units_2}-{self.units_3} {self.opt_name}')
        plt.plot(history.history['val_accuracy'][4:], label=f'test {self.units_1}-{self.units_2}-{self.units_3} {self.opt_name}')
        plt.legend()

        nn_r.save(f'{MODEL_NN_FOLDER_NAME}/{self.model_id}/model.keras')
        self.nn_r = nn_r
        # open text file and write params
        with open(f'{MODEL_NN_FOLDER_NAME}/{self.model_id}/params.txt', "w") as params_file:
            for key, value in self.params.items():
                params_file.write(f'{key} : {value}\n')
        # open text file and write mae performance
        with open(f'{MODEL_NN_FOLDER_NAME}/{self.model_id}/performance.txt', "w") as text_file:
            text_file.write(text)
        self.text = text

    def predict(self, start = 0, end = -1):
        df_data = self.df_data[start:end]
        pred_x_tot = self.nn_r.predict(self.scaler.transform(df_data[self.col_selected]))
        gc.collect()

        df_ori = df_data[self.col_range].reset_index(drop=True)
        y_pred = pd.DataFrame(pred_x_tot, columns=self.col_range)
        df_ori['datetime'] = df_data['datetime'].values
        y_pred['datetime'] = df_data['datetime'].values

        df_ori.reset_index(drop=True, inplace=True)
        y_pred.reset_index(drop=True, inplace=True)
        File.write_df_on_file(df_ori, f'{MODEL_NN_FOLDER_NAME}/{self.model_id}/frg')
        File.write_df_on_file(y_pred, f'{MODEL_NN_FOLDER_NAME}/{self.model_id}/bkg')
        return df_ori, y_pred

    def update_summary(self):
        with open(MODEL_NN_FOLDER_NAME + '/models_params.csv', 'a') as f:
            list_tmp = list(self.params.values()) + self.mae_tr_list
            f.write('\t'.join([str(value) for value in list_tmp] + ['\n']))


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


class MultiMedianKNeighborsRegressor():
    def __init__(self, df_data, col_range, col_selected):
        self.col_range = col_range
        self.col_selected = col_selected
        self.df_data = df_data

        self.y = None
        self.X = None
        self.multi_reg = None

    def create_model(self, n_neighbors=5):
        self.y = self.df_data[self.col_range].astype('float32')
        self.X = self.df_data[self.col_selected].astype('float32')
        self.multi_reg = MultiOutputRegressor(MedianKNeighborsRegressor(n_neighbors=n_neighbors))

    def train(self):
        self.multi_reg.fit(self.X, self.y)

    def predict(self, start = 0, end = -1):
        df_data = self.df_data[start:end]
        df_ori = df_data[self.col_range].reset_index(drop=True)
        y_pred = self.multi_reg.predict(df_data[self.col_selected])
        return df_ori, y_pred
