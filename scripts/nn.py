
import os
from scripts.utils import Logger, logger_decorator
import matplotlib.pyplot as plt
import pandas as pd
import gc
# from datetime import date
import numpy as np
import shap
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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
# Explainability
from tensorflow.keras.utils import plot_model
from scripts.config import MODEL_NN_FOLDER_NAME
# from tensorflow.keras import backend as K
# import tensorflow.keras.losses as losses
# from tensorflow.python.ops import math_ops
# from tensorflow.python.framework import ops


class NN:
    logger = Logger('NN').get_logger()

    @logger_decorator(logger)
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

    @logger_decorator(logger)
    def trim_hyperparams_combinations(self, hyperparams_combinations):
        hyperparams_combinations_tmp = []
        uniques = set()
        model_id = 0
        if os.path.exists(MODEL_NN_FOLDER_NAME + '/models_params.csv'):
            os.remove(MODEL_NN_FOLDER_NAME + '/models_params.csv')
        with open(MODEL_NN_FOLDER_NAME + '/models_params.csv', 'a') as f:
            f.write('\t'.join(['model_id', 'units_1', 'units_2', 'units_3', 'norm', 'drop', 'epochs', 'bs', 'do', 'opt_name', 'lr', 'loss_type', 'top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']) + '\n')
        for units_1, units_2, units_3, norm, drop, epochs, bs, do, opt_name, lr, loss_type in hyperparams_combinations:
            sorted_tuple = tuple([value for value in [units_1, units_2, units_3] if value > 0] + [norm, drop, epochs, bs, do, opt_name, lr, loss_type])
            if len(sorted_tuple) == 8 or sorted_tuple in uniques:
                continue
            else:
                print(sorted_tuple)
                uniques.add(sorted_tuple)
                hyperparams_combinations_tmp.append((model_id, units_1, units_2, units_3, norm, drop, epochs, bs, do, opt_name, lr, loss_type))
                model_id += 1

        return hyperparams_combinations_tmp

    @logger_decorator(logger)
    def use_previous_hyperparams_combinations(self, hyperparams_combinations):
        hyperparams_combinations_tmp = []
        uniques = set()
        model_id = 0
        if os.path.exists(MODEL_NN_FOLDER_NAME):
            first = sorted(os.listdir(MODEL_NN_FOLDER_NAME), key=(lambda x: int(x) if len(x) < 4 else 0))[-1]
        for units_1, units_2, units_3, norm, drop, epochs, bs, do, opt_name, lr, loss_type in hyperparams_combinations:
            sorted_tuple = tuple([value for value in [units_1, units_2, units_3] if value > 0] + [norm, drop, epochs, bs, do, opt_name, lr, loss_type])
            if len(sorted_tuple) == 8 or sorted_tuple in uniques:
                continue
            else:
                print(sorted_tuple)
                uniques.add(sorted_tuple)
                hyperparams_combinations_tmp.append((model_id, units_1, units_2, units_3, norm, drop, epochs, bs, do, opt_name, lr, loss_type))
                model_id += 1
        
        return hyperparams_combinations_tmp[int(first):]

    @logger_decorator(logger)
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

    @logger_decorator(logger)
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

        self.nn_r.compile(loss=self.loss_type, optimizer=opt, metrics=['accuracy'])

    @logger_decorator(logger)
    def scheduler(self, epoch, lr_actual):
        if epoch < 4:
            return self.lr*12.5
        if 4 <= epoch < 12:
            return self.lr*2
        if 12 <= epoch:
            return self.lr/2

    @logger_decorator(logger)
    def train(self):
        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=10, start_from_epoch=50)
        mc = ModelCheckpoint(f'{MODEL_NN_FOLDER_NAME}/{self.model_id}/model.keras', monitor='val_loss', mode='min',
                                verbose=0, save_best_only=True)

        if not self.lr:
            callbacks = [es, mc]
        else:
            call_lr = LearningRateScheduler(self.scheduler)
            callbacks = [es, mc, call_lr]
        history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs,
                        validation_split=0.3, callbacks=callbacks)
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
        return history

    @logger_decorator(logger)
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
        # File.write_df_on_file(df_ori, f'{MODEL_NN_FOLDER_NAME}/{self.model_id}/frg')
        # File.write_df_on_file(y_pred, f'{MODEL_NN_FOLDER_NAME}/{self.model_id}/bkg')
        return df_ori, y_pred

    @logger_decorator(logger)
    def update_summary(self):
        with open(MODEL_NN_FOLDER_NAME + '/models_params.csv', 'a') as f:
            list_tmp = list(self.params.values()) + self.mae_tr_list
            f.write('\t'.join([str(value) for value in list_tmp] + ['\n']))


class MedianKNeighborsRegressor(KNeighborsRegressor):
    logger = Logger('MedianKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
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
    logger = Logger('MultiMedianKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, col_range, col_selected):
        self.col_range = col_range
        self.col_selected = col_selected
        self.df_data = df_data

        self.y = None
        self.X = None
        self.multi_reg = None

    @logger_decorator(logger)
    def create_model(self, n_neighbors=5):
        self.y = self.df_data[self.col_range].astype('float32')
        self.X = self.df_data[self.col_selected].astype('float32')
        self.multi_reg = MultiOutputRegressor(MedianKNeighborsRegressor(n_neighbors=n_neighbors))

    @logger_decorator(logger)
    def train(self):
        self.multi_reg.fit(self.X, self.y)

    @logger_decorator(logger)
    def predict(self, start = 0, end = -1):
        df_data = self.df_data[start:end]
        df_ori = df_data[self.col_range].reset_index(drop=True)
        y_pred = self.multi_reg.predict(df_data[self.col_selected])
        return df_ori, y_pred

class MultiMeanKNeighborsRegressor():
    logger = Logger('MultiMeanKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, col_range, col_selected):
        self.col_range = col_range
        self.col_selected = col_selected
        self.df_data = df_data

        self.y = None
        self.X = None
        self.multi_reg = None

    @logger_decorator(logger)
    def create_model(self, n_neighbors=5):
        self.y = self.df_data[self.col_range].astype('float32')
        self.X = self.df_data[self.col_selected].astype('float32')
        self.multi_reg = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=5)

    @logger_decorator(logger)
    def train(self):
        self.multi_reg.fit(self.X, self.y)

    @logger_decorator(logger)
    def predict(self, start = 0, end = -1):
        df_data = self.df_data[start:end]
        df_ori = df_data[self.col_range].reset_index(drop=True)
        y_pred = self.multi_reg.predict(df_data[self.col_selected])
        return df_ori, pd.DataFrame(y_pred, columns=self.col_range)
