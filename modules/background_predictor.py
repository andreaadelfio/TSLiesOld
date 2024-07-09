'''This module contains the classes for the Neural Network and K-Nearest Neighbors models.'''
import os
import gc
import pandas as pd
import numpy as np
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE, median_absolute_error as MeAE
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
# Keras
from keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras import Input, Model
from keras.layers import Dense, Dropout, BatchNormalization, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.utils import plot_model
# Explainability
import matplotlib.pyplot as plt
# import shap
# ACDAnomalies modules
try:
    import modules.lime.lime_tabular as lime_tabular
    from modules.config import BACKGROUND_PREDICTION_FOLDER_NAME
    from modules.utils import Logger, logger_decorator, File
    from modules.plotter import Plotter
except:
    import lime.lime_tabular as lime_tabular
    from config import BACKGROUND_PREDICTION_FOLDER_NAME
    from utils import Logger, logger_decorator, File
    from plotter import Plotter


def get_feature_importance(model_path, inputs_outputs_df, y_cols, x_cols, num_sample = 100, model = None, show=True, save=True):
    '''Get the feature importance using LIME and SHAP and plots it with matplotlib barh.
    
    Parameters:
    ----------
        model_path (str): The path to the model.
        inputs_outputs_df (pd.DataFrame): The input and output data.
        y_cols (list): The columns of the output data.
        x_cols (list): The columns of the input data.
        show (bool): Whether to show the plot.'''
    X_test = inputs_outputs_df[x_cols]
    if model is None:
        model = load_model(model_path)
        scaler = StandardScaler()
        scaler.fit(X_test)
    else:
        scaler = model.scaler
        model = model.nn_r
    X_test =  pd.DataFrame(scaler.transform(X_test), columns=x_cols)

    y_test = inputs_outputs_df[y_cols]
    X_back = X_test.sample(num_sample)
    importance_dict = {face: {col: 0 for col in X_back.columns} for face in y_cols}
    print(X_back.columns)
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_back),
        feature_names=X_back.columns,
        class_names=list(y_test.columns),
        mode='regression'
    )

    for i in range(num_sample):
        for mylabel, face in enumerate(y_cols):
            exp = explainer.explain_instance(
                data_row=X_back.iloc[i],
                mylabel=mylabel,
                predict_fn=model.predict,
                num_features=len(X_back.columns)
            )
            for feature, weight in exp.as_list(label=mylabel):
                for col in set(x_cols):
                    if col in feature:
                        importance_dict[face][col] += np.abs(weight) / num_sample
                        break

    summed_importance_dict = {col: 0 for col in importance_dict['top'].keys()}
    for face in y_cols:
        for col, value in importance_dict[face].items():
            summed_importance_dict[col] += value

    summed_sorted_importance_dict = dict(sorted(summed_importance_dict.items(), key=lambda item: item[1]))

    sorted_importance_dict = {face: dict(sorted(value.items(), key=lambda item: item[1])) for face, value in importance_dict.items()}
    
    file_name = 'Feature Importance with Lime' if model_path.endswith('.keras') else model_path
    plt.figure(num=file_name, figsize=(10, 8))
    left = {col: 0 for col in importance_dict['top']}
    for i, key in enumerate(importance_dict.keys()):
        sorted_importance_dict[key] = {k: sorted_importance_dict[key][k] for k in summed_sorted_importance_dict}
        left_arr = [left[key] for key in sorted_importance_dict[key]]
        bars = plt.barh(list(sorted_importance_dict[key].keys()), list(sorted_importance_dict[key].values()), left=left_arr, label=key)
        for col in sorted_importance_dict[key]:
            left[col] += sorted_importance_dict[key][col]
    for bar, sum in zip(bars, summed_sorted_importance_dict.values()):
        plt.text(sum * 1.002, bar.get_y() + bar.get_height() / 2, f'{sum:.4f}', va='center', color='grey')
    plt.legend(loc='lower right')
    plt.tight_layout()

    # e = shap.KernelExplainer(model, X_back)
    # shap_values_orig = e(X_back)
    # shap_values_sum = np.mean(shap_values_orig.values, axis=2)
    # shap_values_sum = shap.Explanation(values=shap_values_sum,
    #                                 base_values=shap_values_orig.base_values[0],
    #                                 data=shap_values_orig.data,
    #                                 feature_names=shap_values_orig.feature_names)
    # plt.figure(num='Feature Importance with SHAP')
    # shap.plots.bar(shap_values_sum, max_display=len(X_back.columns), show=False)

    # idx = 0
    # plt.figure(num='Waterfall plot with SHAP')
    # shap.plots._waterfall.waterfall_legacy(e.expected_value[idx], shap_values_orig[0].T[idx],
    #                                        feature_names=x_cols, max_display=len(x_cols))

    if show:
        plt.show()
    if save:
        Plotter.save(os.path.dirname(model_path) if model_path.endswith('.keras') else model_path)

class MLObject:
    '''The class used to handle the Machine Learning.'''
    def __init__(self):
        self.nn_r = None
        self.scaler = None
        self.model_path = None

    def set_model(self, model_path: str):
        '''Sets the model from the model path.
        
        Parameters:
        ----------
            model_path (str): The path to the model.
            
        Returns:
        --------
            Model: The model.'''
        self.model_path = model_path
        self.nn_r = load_model(model_path)
        return self.nn_r

    def set_scaler(self, train: pd.DataFrame):
        '''Sets the scaler for the model.
        
        Parameters:
        ----------
            train (pd.DataFrame): The training data.'''
        scaler = StandardScaler()
        scaler.fit(train)
        self.scaler = scaler


class FFNNPredictor(MLObject):
    '''The class for the Neural Network model.'''
    logger = Logger('NN').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols):
        self.y_cols = y_cols
        self.x_cols = x_cols
        self.df_data = df_data

        if not os.path.exists(f'{BACKGROUND_PREDICTION_FOLDER_NAME}'):
            os.makedirs(f'{BACKGROUND_PREDICTION_FOLDER_NAME}')
        self.y = None
        self.X = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.scaler = None
        self.nn_r = None
        self.text = None
        self.model_id = None
        self.model_path = None
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
        '''Trims the hyperparameters combinations to avoid duplicates.'''
        hyperparams_combinations_tmp = []
        uniques = set()
        model_id = 0
        if os.path.exists(BACKGROUND_PREDICTION_FOLDER_NAME + '/models_params.csv'):
            os.remove(BACKGROUND_PREDICTION_FOLDER_NAME + '/models_params.csv')
        with open(BACKGROUND_PREDICTION_FOLDER_NAME + '/models_params.csv', 'a') as f:
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
    def use_previous_hyperparams_combinations(self, hyperparams_combinations) -> list:
        '''Uses the previous hyperparameters combinations.
        
        Parameters:
        ----------
            hyperparams_combinations (list): The hyperparameters combinations.
            
        Returns:
        --------
            list: The hyperparameters combinations.'''
        hyperparams_combinations_tmp = []
        uniques = set()
        model_id = 0
        if os.path.exists(BACKGROUND_PREDICTION_FOLDER_NAME):
            first = sorted(os.listdir(BACKGROUND_PREDICTION_FOLDER_NAME), key=(lambda x: int(x) if len(x) < 4 else 0))[-1]
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
        '''Sets the hyperparameters for the model.
        
        Parameters:
        ----------
            params (dict): The hyperparameters.
            
        Example::

            params = {'model_id': 0, 'units_1': 128, 'units_2': 128, 'units_3': 128,
                      'norm': True, 'drop': True, 'epochs': 100, 'bs': 32, 'do': 0.5,
                      'opt_name': 'Adam', 'lr': 0.001, 'loss_type': 'mean_squared_error'}
            nn.set_hyperparams(params)'''
        self.params = params
        self.model_id = params['model_id']
        if not os.path.exists(f'{BACKGROUND_PREDICTION_FOLDER_NAME}/{self.model_id}'):
            os.makedirs(f'{BACKGROUND_PREDICTION_FOLDER_NAME}/{self.model_id}')
        self.model_path = f'{BACKGROUND_PREDICTION_FOLDER_NAME}/{self.model_id}/model.keras'
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
        '''Creates the model.'''
        self.y = self.df_data[self.y_cols].astype('float32')
        self.X = self.df_data[self.x_cols].astype('float32')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=0, shuffle=True)

        self.set_scaler(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

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
        outputs = Dense(len(self.y_cols), activation='linear')(hidden)

        self.nn_r = Model(inputs=[inputs], outputs=outputs)
        plot_model(self.nn_r, to_file=f'{BACKGROUND_PREDICTION_FOLDER_NAME}/{self.model_id}/schema.png',
                   show_shapes=True, show_layer_names=True, rankdir='LR')

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
        '''The learning rate scheduler.'''
        if epoch < 0.06 * self.epochs:
            return self.lr*12.5
        if 0.06 * self.epochs <= epoch < 0.20 * self.epochs:
            return self.lr*2
        if 0.20 * self.epochs <= epoch:
            return self.lr/2

    @logger_decorator(logger)
    def train(self):
        '''Trains the model.'''
        
        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, 
                           patience=10, start_from_epoch=80)
        mc = ModelCheckpoint(self.model_path, 
                             monitor='val_loss', mode='min', verbose=0, save_best_only=True)

        if not self.lr:
            callbacks = [es, mc]
        else:
            call_lr = LearningRateScheduler(self.scheduler)
            callbacks = [es, mc, call_lr]
        history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs,
                        validation_split=0.3, callbacks=callbacks)
        
        nn_r = load_model(self.model_path)

        pred_train = nn_r.predict(self.X_train)
        pred_test = nn_r.predict(self.X_test)
        idx = 0
        text = ''
        for col in self.y_cols:
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

        nn_r.save(self.model_path)
        self.nn_r = nn_r
        with open(f'{BACKGROUND_PREDICTION_FOLDER_NAME}/{self.model_id}/params.txt', "w") as params_file:
            for key, value in self.params.items():
                params_file.write(f'{key} : {value}\n')
        with open(f'{BACKGROUND_PREDICTION_FOLDER_NAME}/{self.model_id}/performance.txt', "w") as text_file:
            text_file.write(text)
        self.text = text
        return history

    @logger_decorator(logger)
    def predict(self, start = 0, end = -1, write=True):
        '''Predicts the output data.
        
        Parameters:
        ----------
            start (int): The starting index. Default is 0.
            end (int): The ending index. Defualt is -1.'''
        df_data = self.df_data[start:end]
        pred_x_tot = self.nn_r.predict(self.scaler.transform(df_data[self.x_cols]))
        gc.collect()

        df_ori = df_data[self.y_cols].reset_index(drop=True)
        y_pred = pd.DataFrame(pred_x_tot, columns=self.y_cols)
        df_ori['datetime'] = df_data['datetime'].values
        y_pred['datetime'] = df_data['datetime'].values

        df_ori.reset_index(drop=True, inplace=True)
        y_pred.reset_index(drop=True, inplace=True)
        if write:
            path = f'{BACKGROUND_PREDICTION_FOLDER_NAME}/{self.model_id}'
            if not self.model_id:
                path = os.path.dirname(self.model_path)
            File.write_df_on_file(df_ori, f'{path}/frg')
            File.write_df_on_file(y_pred, f'{path}/bkg')
        return df_ori, y_pred

    @logger_decorator(logger)
    def update_summary(self):
        '''Updates the summary file with the model parameters'''
        with open(BACKGROUND_PREDICTION_FOLDER_NAME + '/models_params.csv', 'a') as f:
            list_tmp = list(self.params.values()) + self.mae_tr_list
            f.write('\t'.join([str(value) for value in list_tmp] + ['\n']))


class RNNPredictor(FFNNPredictor):
    def __init__(self, df_data, y_cols, x_cols, timestep):
        super().__init__(df_data, y_cols, x_cols)
        self.timesteps = timestep

    # @logger_decorator(logger)
    def create_model(self):
        '''Creates the model.'''
        self.y = self.df_data[self.y_cols].astype('float32')
        self.X = self.df_data[self.x_cols].astype('float32')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X.values, self.y.values, test_size=0.25, random_state=0, shuffle=True)

        self.set_scaler(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.y_train, self.y_test = self.y_train[self.timesteps:], self.y_test[self.timesteps:]
        X_train = np.array([self.X_train[i:i + self.timesteps] for i in np.arange(len(self.X_train) - self.timesteps)])
        self.X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        X_test = np.array([self.X_test[i:i + self.timesteps] for i in np.arange(len(self.X_test) - self.timesteps)])
        self.X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
        
        inputs = Input(shape=(None, len(self.x_cols)))  # None indica un numero variabile di timestep

        if self.units_1:
            hidden = LSTM(self.units_1, return_sequences=True)(inputs)  # return_sequences=True se vuoi collegare più layer LSTM
            if self.norm:
                hidden = BatchNormalization()(hidden)
            if self.drop:
                hidden = Dropout(self.do)(hidden)
        else:
            hidden = inputs

        if self.units_2:
            hidden = LSTM(self.units_2, return_sequences=True)(hidden)  # Assicurati che return_sequences sia True se aggiungi altri layer LSTM dopo questo
            if self.norm:
                hidden = BatchNormalization()(hidden)
            if self.drop:
                hidden = Dropout(self.do)(hidden)

        if self.units_3:
            hidden = LSTM(self.units_3)(hidden)  # return_sequences=False (default) se questo è l'ultimo layer LSTM
            if self.norm:
                hidden = BatchNormalization()(hidden)
            if self.drop:
                Dropout(self.do)(hidden)

        outputs = Dense(len(self.y_cols), activation='linear')(hidden)

        self.nn_r = Model(inputs=[inputs], outputs=outputs)
        plot_model(self.nn_r, to_file=f'{BACKGROUND_PREDICTION_FOLDER_NAME}/{self.model_id}/schema.png',
                   show_shapes=True, show_layer_names=True, rankdir='LR')

        if self.opt_name == 'Adam':
            opt = Adam(beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        elif self.opt_name == 'Nadam':
            opt = Nadam(beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        elif self.opt_name == 'RMSprop':
            opt = RMSprop(rho=0.6, momentum=0.0, epsilon=1e-07)
        elif self.opt_name == 'SGD':
            opt = SGD()

        self.nn_r.compile(loss=self.loss_type, optimizer=opt, metrics=['accuracy'])

    # @logger_decorator(logger)
    def train(self):
        '''Trains the model.'''
        
        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, 
                           patience=10, start_from_epoch=80)
        mc = ModelCheckpoint(self.model_path, 
                             monitor='val_loss', mode='min', verbose=0, save_best_only=True)

        if not self.lr:
            callbacks = [es, mc]
        else:
            call_lr = LearningRateScheduler(self.scheduler)
            callbacks = [es, mc, call_lr]
        history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs,
                        validation_split=0.3, callbacks=callbacks)
        
        nn_r = load_model(self.model_path)

        pred_train = nn_r.predict(self.X_train)
        pred_test = nn_r.predict(self.X_test)
        idx = 0
        text = ''
        for col in self.y_cols:
            mae_tr = MAE(self.y_train[:, idx], pred_train[:, idx])
            self.mae_tr_list.append(mae_tr)
            mae_te = MAE(self.y_test[:, idx], pred_test[:, idx])
            diff_i = (self.y_test[:, idx] - pred_test[:, idx])
            mean_diff_i = (diff_i).mean()
            meae_tr = MeAE(self.y_train[:, idx], pred_train[:, idx])
            meae_te = MeAE(self.y_test[:, idx], pred_test[:, idx])
            median_diff_i = np.median(diff_i)
            text += f"MAE_train_{col} : {mae_tr:0.5f}\t" + \
                    f"MAE_test_{col} : {mae_te:0.5f}\t" + \
                    f"mean_diff_test_pred_{col} : {mean_diff_i:0.5f}\t" + \
                    f"MeAE_train_{col} {meae_tr:0.5f}\t" + \
                    f"MeAE_test_{col} {meae_te:0.5f}\t" + \
                    f"median_diff_test_pred_{col} {median_diff_i:0.5f}\n"
            idx = idx + 1

        nn_r.save(self.model_path)
        self.nn_r = nn_r
        with open(f'{BACKGROUND_PREDICTION_FOLDER_NAME}/{self.model_id}/params.txt', "w") as params_file:
            for key, value in self.params.items():
                params_file.write(f'{key} : {value}\n')
        with open(f'{BACKGROUND_PREDICTION_FOLDER_NAME}/{self.model_id}/performance.txt', "w") as text_file:
            text_file.write(text)
        self.text = text
        return history

    # @logger_decorator(logger)
    def predict(self, start = 0, end = -1, write=True, batched=False):
        '''Predicts the output data.
        
        Parameters:
        ----------
            start (int): The starting index. Default is 0.
            end (int): The ending index. Defualt is -1.'''
        data_to_be_scaled = self.df_data[self.x_cols][start:end]
        data = self.scaler.transform(data_to_be_scaled)
        data = np.array([data[i:i + self.timesteps] for i in np.arange(len(data) - self.timesteps)])
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))
        if batched:
            pred_x_tot = np.array([])
            batch_size = len(data)//self.timesteps
            for i in range(0, len(data), batch_size):
                pred_x_tot = np.append(pred_x_tot, self.nn_r.predict(data[i:i + batch_size]))
            pred_x_tot = np.reshape(pred_x_tot, (len(data), len(self.y_cols)))
        else:
            pred_x_tot = self.nn_r.predict(data)
        gc.collect()

        df_ori = self.df_data[start:end][self.y_cols].reset_index(drop=True)
        y_pred = pd.DataFrame(pred_x_tot, columns=self.y_cols)
        print(len(y_pred), len(self.df_data[start:end]['datetime'].values))
        df_ori['datetime'] = self.df_data[start:end]['datetime'].values
        y_pred['datetime'] = self.df_data[start+self.timesteps:end]['datetime'].values

        df_ori.reset_index(drop=True, inplace=True)
        y_pred.reset_index(drop=True, inplace=True)
        print(len(y_pred))
        if write:
            path = f'{BACKGROUND_PREDICTION_FOLDER_NAME}/{self.model_id}'
            if not self.model_id:
                path = os.path.dirname(self.model_path)
            File.write_df_on_file(df_ori, f'{path}/frg')
            File.write_df_on_file(y_pred, f'{path}/bkg')
        return df_ori, y_pred
        

class MedianKNeighborsRegressor(KNeighborsRegressor):
    '''The class for the Median K-Nearest Neighbors model.'''
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
    '''The class for the Multi Median K-Nearest Neighbors model.'''
    logger = Logger('MultiMedianKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols):
        self.y_cols = y_cols
        self.x_cols = x_cols
        self.df_data = df_data

        self.y = None
        self.X = None
        self.multi_reg = None

    @logger_decorator(logger)
    def create_model(self, n_neighbors=5):
        self.y = self.df_data[self.y_cols].astype('float32')
        self.X = self.df_data[self.x_cols].astype('float32')
        self.multi_reg = MultiOutputRegressor(MedianKNeighborsRegressor(n_neighbors=n_neighbors))

    @logger_decorator(logger)
    def train(self):
        self.multi_reg.fit(self.X, self.y)

    @logger_decorator(logger)
    def predict(self, start = 0, end = -1):
        df_data = self.df_data[start:end]
        df_ori = df_data[self.y_cols].reset_index(drop=True)
        y_pred = self.multi_reg.predict(df_data[self.x_cols])
        return df_ori, y_pred

class MultiMeanKNeighborsRegressor():
    logger = Logger('MultiMeanKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols):
        self.y_cols = y_cols
        self.x_cols = x_cols
        self.df_data = df_data

        self.y = None
        self.X = None
        self.multi_reg = None

    @logger_decorator(logger)
    def create_model(self, n_neighbors=5):
        self.y = self.df_data[self.y_cols].astype('float32')
        self.X = self.df_data[self.x_cols].astype('float32')
        self.multi_reg = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=5)

    @logger_decorator(logger)
    def train(self):
        self.multi_reg.fit(self.X, self.y)

    @logger_decorator(logger)
    def predict(self, start = 0, end = -1):
        df_data = self.df_data[start:end]
        df_ori = df_data[self.y_cols].reset_index(drop=True)
        y_pred = self.multi_reg.predict(df_data[self.x_cols])
        return df_ori, pd.DataFrame(y_pred, columns=self.y_cols)

if __name__ == '__main__':
    inputs_outputs_df = File.read_dfs_from_pk_folder()

    y_cols = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
    y_smooth_cols = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']
    x_cols = [col for col in inputs_outputs_df.columns if col not in y_cols + y_smooth_cols + ['datetime']]
    MODEL_PATH = './data/model_nn/0/model.keras'
    get_feature_importance(MODEL_PATH, inputs_outputs_df, y_cols, x_cols, num_sample=10, show=True)