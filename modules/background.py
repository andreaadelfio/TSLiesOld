'''
This module contains the classes for the Neural Network and K-Nearest Neighbors models.
'''
import itertools
import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE, median_absolute_error as MeAE
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
# Keras
from keras.optimizers import Adam, Nadam, RMSprop, SGD # pylint: disable=E0401
from keras import Input, Model
from keras.layers import Dense, Dropout, BatchNormalization, LSTM # pylint: disable=E0401
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, LambdaCallback # pylint: disable=E0401
from keras.models import load_model # pylint: disable=E0401
from keras.utils import plot_model # pylint: disable=E0401
# TensorFlow for Bayesian Neural Network
import tf_keras
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
import tensorflow as tf
# ACDAnomalies modules
try:
    from modules.config import BACKGROUND_PREDICTION_FOLDER_NAME, DIR
    from modules.utils import Logger, logger_decorator, File, Data
    from modules.plotter import Plotter
except:
    from config import BACKGROUND_PREDICTION_FOLDER_NAME, DIR
    from utils import Logger, logger_decorator, File, Data
    from plotter import Plotter


class MLObject:
    '''The class used to handle Machine Learning.'''
    def __init__(self, df_data, y_cols, x_cols, y_cols_raw, y_pred_cols, with_generator=False):
        self.model_name = self.__class__.__name__
        self.training_date = pd.Timestamp.date(pd.Timestamp.now()).strftime('%Y-%m-%d')
        print(f'{self.model_name} - {self.training_date}')
        self.with_generator = with_generator
        self.y_cols = y_cols
        self.x_cols = x_cols
        print(f'x_cols: {x_cols}')
        print(f'y_cols: {y_cols}')
        self.y_cols_raw = y_cols_raw
        self.y_pred_cols = y_pred_cols

        if with_generator: # if the data is too large, use tensorflow DataGenerator
            # to be implemented
            pass
        else:
            self.df_data: pd.DataFrame = df_data
        self.y = self.df_data[self.y_cols].astype('float32')
        self.X = self.df_data[self.x_cols].astype('float32')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42, shuffle=True)
        
        self.set_scaler(self.X)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        self.model_path = os.path.join(BACKGROUND_PREDICTION_FOLDER_NAME, self.training_date, self.model_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.nn_r = None
        self.text = None
        self.model_id = None
        self.params = None
        self.norm = None
        self.drop = None
        self.units_for_layers = None
        self.bs = None
        self.do = None
        self.opt_name = None
        self.lr = None
        self.loss_type = None
        self.epochs = None
        self.mae_tr_list = None

    def get_hyperparams_combinations(self, hyperparams_combinations:dict, use_previous:bool=False) -> list:
        '''Trims the hyperparameters combinations to avoid training duplicate models.
        
            Parameters:
            ----------
                hyperparams_combinations (dict): The hyperparameters combinations.
                use_previous (bool): Whether to use the previous hyperparameters combinations found in `BACKGROUND_PREDICTION_FOLDER_NAME`.
                
            Returns:
            --------
                list: The hyperparameters combinations.'''
        hyperparams_combinations_tmp = []
        uniques = set()
        model_id = 0
        first = 0
        if not use_previous:
            model_params_path = os.path.join(self.model_path, 'models_params.csv')
            if os.path.exists(model_params_path):
                os.remove(model_params_path)
            with open(model_params_path, 'w') as f:
                f.write('\t'.join(['model_id'] + list(hyperparams_combinations.keys()) + ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']) + '\n')
        else:
            first = int(sorted(os.listdir(self.model_path), key=(lambda x: int(x) if len(x) < 4 else 0))[-1])

        units_for_layers = list(itertools.product(*hyperparams_combinations['units_for_layers']))
        other_params = [hyperparams_combinations[key] for key in hyperparams_combinations if key != 'units_for_layers']
        all_combinations = list(itertools.product(units_for_layers, *other_params))

        for combination in all_combinations:
            sorted_tuple = tuple([value for value in combination[0] if value and value > 0] + list(combination[1:]))
            if len(sorted_tuple) == 8 or sorted_tuple in uniques:
                continue
            else:
                uniques.add(sorted_tuple)
                hyperparams_dict = {key: value for key, value in zip(hyperparams_combinations.keys(), combination)}
                hyperparams_dict['model_id'] = model_id
                hyperparams_combinations_tmp.append(hyperparams_dict)
                model_id += 1

        return hyperparams_combinations_tmp[int(first):]

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
        self.params['model_name'] = self.model_name
        self.params['training_date'] = self.training_date
        self.params['x_cols'] = self.x_cols
        self.params['y_cols'] = self.y_cols
        self.model_id = params['model_id']
        self.model_path = os.path.join(BACKGROUND_PREDICTION_FOLDER_NAME, self.training_date, self.model_name, str(self.model_id))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.params['model_path'] = self.model_path = os.path.join(self.model_path, 'model.keras')
        self.units_for_layers = params['units_for_layers']
        self.bs = params['bs']
        self.do = params['do']
        self.opt_name = params['opt_name']
        self.lr = params['lr']
        self.loss_type = params['loss_type']
        self.epochs = params['epochs']
        self.norm = params['norm']
        self.drop = params['drop']
        self.mae_tr_list = []

    def set_model(self, model_path: str):
        '''Sets the model from the model path.
        
        Parameters:
        ----------
            model_path (str): The path to the model.
            
        Returns:
        --------
            Model: The model.'''
        self.model_path = os.path.join(DIR, model_path)
        self.nn_r = load_model(self.model_path)
        return self.nn_r

    def set_scaler(self, train: pd.DataFrame = None):
        '''Sets the scaler for the model.
        
        Parameters:
        ----------
            train (pd.DataFrame): The training data.'''
        if train is None:
            train = self.df_data[self.x_cols]
        self.scaler = StandardScaler()
        self.scaler.fit(train)

    def scheduler(self, epoch, lr_actual):
        '''The learning rate scheduler.'''
        # if epoch < 0.06 * self.epochs:
        #     return self.lr*12.5
        # if 0.06 * self.epochs <= epoch < 0.20 * self.epochs:
        #     return self.lr*2
        # if 0.20 * self.epochs:
        #     return self.lr/3
        return lr_actual
        
    def update_summary(self):
        '''Updates the summary file with the model parameters'''
        with open(os.path.join(BACKGROUND_PREDICTION_FOLDER_NAME, self.training_date, self.model_name, 'models_params.csv'), 'a') as f:
            list_tmp = list(self.params.values()) + self.mae_tr_list
            f.write('\t'.join([str(value) for value in list_tmp] + ['\n']))

    def negative_log_likelihood(self, y_true, y_pred):
        '''The negative log likelihood loss function computed using the mean and variance.'''
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        precision = tf.exp(-log_var)
        nll = 0.5 * tf.reduce_mean(log_var + precision * (y_true - mean) ** 2, axis=0)  # Per output
        return tf.reduce_sum(nll)  # Sum over all outputs

    def mae(self, y_true, y_pred):
        '''The mean absolute error metric computed using the mean of the output.'''
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        return tf.reduce_mean(tf.abs(y_true - mean))

    def NLL(self, y_true, y_pred):
        '''The negative log likelihood loss function computed using the mean and variance.'''
        return -tf.reduce_mean(y_pred.log_prob(y_true))

    def MSE(self, y_true, y_pred):
        '''The mean squared error loss function computed using the mean of the output.'''
        return tf.reduce_mean(tf.square(y_true - y_pred.mean()))

    def MAE(self, y_true, y_pred):
        '''The mean absolute error loss function computed using the mean of the output.'''
        return tf.reduce_mean(tf.abs(y_true - y_pred.mean()))

    def KL_divergence(self, posterior, prior):
        '''The Kullback-Leibler divergence loss function.'''
        return tfp.distributions.kl_divergence(posterior, prior)

    def combined_loss(self, y_true, y_pred):
        '''A combined loss function.'''
        nll_loss = self.NLL(y_true, y_pred)
        mae_loss = self.MAE(y_true, y_pred)
        return nll_loss

    class custom_callback(tf_keras.callbacks.Callback):
        '''Custom callback class to end the model training and plot the predictions.'''
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor

        def on_epoch_end(self, epoch, logs={}):
            if (epoch + 1) % 5 == 0:
                self.predictor.predict(start=0, end=1000, mask_column='index', write_bkg=False, save_predictions_plot=True)
            if( logs['mae'] <= 0.003 and logs['accuracy'] > 0.65 ):
                self.model.stop_training = True

    def save_predictions_plots(self, tiles_df, start, end, params):
        '''Saves the prediction plots.'''
        Plotter(df=tiles_df, label='tiles').df_plot_tiles(self.y_cols, x_col='datetime', init_marker=',',
                                                        show=False, smoothing_key='pred')
        for col in self.y_cols_raw:
            Plotter().plot_tile(tiles_df, face=col, smoothing_key = 'pred')
        Plotter().plot_pred_true(tiles_df, self.y_pred_cols, self.y_cols_raw)
        Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME, params, (start, end))

class FFNNPredictor(MLObject):
    '''The class for the Feed Forward Neural Network model.'''
    logger = Logger('FFNNPredictor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_cols_raw, y_pred_cols, with_generator=False):
        super().__init__(df_data, y_cols, x_cols, y_cols_raw, y_pred_cols, with_generator)

    @logger_decorator(logger)
    def create_model(self):
        '''Creates the model.'''
        inputs = Input(shape=(self.X_train.shape[1],))
        for count, units in enumerate(list(self.units_for_layers)):
            hidden = Dense(units, activation='relu')(inputs if count == 0 else hidden)
            if self.norm:
                hidden = BatchNormalization()(hidden)
            if self.drop:
                hidden = Dropout(self.do)(hidden)
        outputs = Dense(len(self.y_cols), activation='linear')(hidden)

        self.nn_r = Model(inputs=[inputs], outputs=outputs)
        plot_model(self.nn_r, to_file=os.path.join(os.path.dirname(self.model_path), 'schema.png'),
                   show_shapes=True, show_layer_names=True, rankdir='TB')

        opt = None
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
    def train(self):
        '''Trains the model.'''
        
        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.002, 
                           patience=10, start_from_epoch=190)
        mc = ModelCheckpoint(self.model_path, 
                             monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        # batch_print_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.predict(start='2024-05-08 20:30:00', end='2024-05-08 23:40:00', write_bkg=False, save_predictions_plot=True))


        if not self.lr:
            callbacks = [es, mc]
        else:
            call_lr = LearningRateScheduler(self.scheduler)
            callbacks = [es, mc, call_lr]

        if self.with_generator:
            history = self.nn_r.fit(self.df_data, epochs=self.epochs, batch_size=32, validation_split=0.3, callbacks=callbacks)
        else:
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
        with open(os.path.join(os.path.dirname(self.model_path), 'params.txt'), "w") as params_file:
            for key, value in self.params.items():
                params_file.write(f'{key} : {value}\n')
        with open(os.path.join(os.path.dirname(self.model_path), 'performance.txt'), "w") as text_file:
            text_file.write(text)
        self.text = text
        return history

    @logger_decorator(logger)
    def predict(self, start:str|int = 0, end:str|int = -1, mask_column='index', write_bkg=True, write_frg=False, batch_size=1, save_predictions_plot=False, support_variables=[]) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''Predicts the output data.
        
        Parameters:
        ----------
            start (str|int): The starting index. Default is 0.
            end (str|int): The ending index. Defualt is -1.
            mask_column (str): The column to mask the data.
            write_bkg (bool): Whether to write the background data. Default is `True`.
            write_frg (bool): Whether to write the foreground data. Default is `False`.
            batch_size (int): The batch size for the prediction. Default is 1.
            save_predictions_plot (bool): Whether to save the predictions plot. Default is `False`.
            support_variables (list): The support variables to plot.
        '''
        df_data = Data.get_masked_dataframe(data=self.df_data, start=start, stop=end, column=mask_column, reset_index=False)
        if df_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        scaled_data = self.scaler.transform(df_data[self.x_cols])
        if batch_size > 1:
            pred_x_tot = np.array([])
            for i in range(0, len(scaled_data), batch_size):
                pred_x_tot = np.append(pred_x_tot, self.nn_r.predict(scaled_data[i:i + batch_size]))
        else:
            pred_x_tot = self.nn_r.predict(scaled_data)
        y_pred = pd.DataFrame(pred_x_tot, columns=self.y_cols)
        y_pred['datetime'] = df_data['datetime'].values
        y_pred.reset_index(drop=True, inplace=True)
        df_ori = df_data[self.y_cols].reset_index(drop=True)
        df_ori['datetime'] = df_data['datetime'].values
        df_ori.reset_index(drop=True, inplace=True)
        if write_bkg:
            path = os.path.join(os.path.dirname(self.model_path))
            if not self.model_id:
                path = os.path.dirname(self.model_path)
            File.write_df_on_file(y_pred, os.path.join(path, 'bkg'))
            gc.collect()

            if write_frg:
                path = os.path.join(os.path.dirname(self.model_path))
                if not self.model_id:
                    path = os.path.dirname(self.model_path)
                File.write_df_on_file(df_ori, os.path.join(path, 'frg'))
        if save_predictions_plot:
            y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(self.y_pred_cols, self.y_cols)}).drop(columns=self.y_cols)
            tiles_df = Data.merge_dfs(df_data[self.y_cols_raw + ['datetime'] + support_variables], y_pred)
            self.save_predictions_plots(tiles_df, start, end, self.params)
        return df_ori, y_pred

class PBNNPredictor(MLObject):
    '''The class for the Probabilistic Bayesian Neural Network model.'''
    logger = Logger('PBNNPredictor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_cols_raw, y_pred_cols, with_generator=False):
        super().__init__(df_data, y_cols, x_cols, y_cols_raw, y_pred_cols, with_generator)
    
    def prior_trainable(self, kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf_keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=1),
                reinterpreted_batch_ndims=1)),
        ])

    def random_gaussian_initializer(self, shape, dtype):
        n = int(shape / 2)
        loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
        loc = tf.Variable(
            initial_value=loc_norm(shape=(n,), dtype=dtype)
        )
        scale_norm = tf.random_normal_initializer(mean=-3., stddev=0.1)
        scale = tf.Variable(
            initial_value=scale_norm(shape=(n,), dtype=dtype)
        )
        return tf.concat([loc, scale], 0)
    
    # The posterior is modeled as n_weights independent Normal distribution with learnable parameters
    def posterior_mean_field(self, kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf_keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype, initializer=self.random_gaussian_initializer, trainable=True),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                            scale=1e-5 + 0.001*tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])
    
    def normal_sp(self, params):
        return tfd.Normal(loc=params[:,:len(self.y_cols)],\
                      scale=1e-5 + 0.00001*tf_keras.backend.exp(params[:,len(self.y_cols):]))# both parameters are learnable

    @logger_decorator(logger)
    def create_model(self):
        '''Builds the Bayesian Neural Network model.'''

        self.nn_r = tf_keras.Sequential([
            tf_keras.Input(shape=(len(self.x_cols), )),
        ])

        for units in list(self.units_for_layers):
            self.nn_r.add(tfpl.DenseVariational(units, self.posterior_mean_field, self.prior_trainable, kl_weight=1/self.X_train.shape[0]))
        self.nn_r.add(tfpl.DenseVariational(2*len(self.y_cols), self.posterior_mean_field, self.prior_trainable, kl_weight=1/self.X_train.shape[0]))
        self.nn_r.add(tfpl.DistributionLambda(self.normal_sp))

        if self.lr:
            opt = tf_keras.optimizers.Adam(learning_rate=self.lr)
        else:
            opt = tf_keras.optimizers.Adam()

        self.nn_r.compile(optimizer=opt,
                      loss=self.combined_loss,
                      metrics=['mae', 'accuracy'])
    
    @logger_decorator(logger)
    def train(self):
        '''Trains the model.'''
        es = tf_keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.002,
                           patience=10, start_from_epoch=190)
        mc = tf_keras.callbacks.ModelCheckpoint(self.model_path,
                             monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        
        call_lr = tf_keras.callbacks.LearningRateScheduler(self.scheduler)
        
        callbacks = [self.custom_callback(self), call_lr]

        if self.with_generator:
            history = self.nn_r.fit(self.df_data, epochs=self.epochs, batch_size=32, validation_split=0.3)
        else:
            history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs, validation_split=0.3,
                      callbacks=callbacks)
        

        # pred_train = nn_r.predict(self.X_train)
        # pred_test = nn_r.predict(self.X_test)
        # idx = 0
        # text = ''
        # for col in self.y_cols:
        #     mae_tr = MAE(self.y_train.iloc[:, idx], pred_train[:, idx])
        #     self.mae_tr_list.append(mae_tr)
        #     mae_te = MAE(self.y_test.iloc[:, idx], pred_test[:, idx])
        #     diff_i = (self.y_test.iloc[:, idx] - pred_test[:, idx])
        #     mean_diff_i = (diff_i).mean()
        #     meae_tr = MeAE(self.y_train.iloc[:, idx], pred_train[:, idx])
        #     meae_te = MeAE(self.y_test.iloc[:, idx], pred_test[:, idx])
        #     median_diff_i = (diff_i).median()
        #     text += f"MAE_train_{col} : {mae_tr:0.5f}\t" + \
        #             f"MAE_test_{col} : {mae_te:0.5f}\t" + \
        #             f"mean_diff_test_pred_{col} : {mean_diff_i:0.5f}\t" + \
        #             f"MeAE_train_{col} {meae_tr:0.5f}\t" + \
        #             f"MeAE_test_{col} {meae_te:0.5f}\t" + \
        #             f"median_diff_test_pred_{col} {median_diff_i:0.5f}\n"
        #     idx = idx + 1

        # nn_r.save(self.model_path)
        with open(os.path.join(os.path.dirname(self.model_path), 'params.txt'), "w") as params_file:
            for key, value in self.params.items():
                params_file.write(f'{key} : {value}\n')
        # with open(os.path.join(os.path.dirname(self.model_path), 'performance.txt'), "w") as text_file:
        #     text_file.write(text)
        # self.text = text
        return history
    
    @logger_decorator(logger)
    def predict(self, start = 0, end = -1, runs=250, mask_column='index', write_bkg=True, write_frg=False, batch_size=1, save_predictions_plot=True, support_variables=[]) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''Predicts the output data.
        
        Parameters:
        ----------
            start (int): The starting index. Default is 0.
            end (int): The ending index. Defualt is -1.'''
        df_data = Data.get_masked_dataframe(data=self.df_data, start=start, stop=end, column=mask_column, reset_index=False)
        if df_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        scaled_data = self.scaler.transform(df_data[self.x_cols])
        if batch_size > 1:
            preds = np.array([])
            for i in range(0, len(scaled_data), batch_size):
                preds = np.append(preds, self.nn_r.predict(scaled_data[i:i + batch_size]))
        else:
            preds = tf.zeros([runs, len(scaled_data), len(self.y_cols)]).numpy()
            for i in tqdm(range(0, runs)):
                preds[i,:] = self.nn_r.predict(scaled_data, verbose=0)
                
        preds_std = tf.experimental.numpy.std( preds, axis=0, keepdims=None ).numpy()
        preds_mean = tf.experimental.numpy.mean( preds, axis=0, dtype=None, out=None, keepdims=None ).numpy()
        y_pred = pd.DataFrame(preds_mean, columns=self.y_cols)
        y_std = pd.DataFrame(preds_std, columns=[f'{col}_std' for col in self.y_cols])
        y_pred = pd.concat([y_pred, y_std], axis=1)
        y_pred['datetime'] = df_data['datetime'].values
        y_pred.reset_index(drop=True, inplace=True)
        df_ori = df_data[self.y_cols].reset_index(drop=True)
        df_ori['datetime'] = df_data['datetime'].values
        df_ori.reset_index(drop=True, inplace=True)
        if write_bkg:
            path = os.path.join(os.path.dirname(self.model_path))
            if not self.model_id:
                path = os.path.dirname(self.model_path)
            File.write_df_on_file(y_pred, os.path.join(path, 'bkg'))
            gc.collect()

            if write_frg:
                path = os.path.join(os.path.dirname(self.model_path))
                if not self.model_id:
                    path = os.path.dirname(self.model_path)
                File.write_df_on_file(df_ori, os.path.join(path, 'frg'))
        if save_predictions_plot:
            y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(self.y_pred_cols, self.y_cols)}).drop(columns=self.y_cols)
            tiles_df = Data.merge_dfs(df_data[self.y_cols_raw + ['datetime'] + support_variables], y_pred)
            self.save_predictions_plots(tiles_df, start, end, self.params)
        return df_ori, y_pred

class BNNPredictor(MLObject):
    '''The class for the Bayesian Neural Network model.'''
    logger = Logger('BNNPredictor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_cols_raw, y_pred_cols, with_generator=False):
        super().__init__(df_data, y_cols, x_cols, y_cols_raw, y_pred_cols, with_generator)
    
    @logger_decorator(logger)
    def create_model(self):
        '''Builds the Bayesian Neural Network model.'''

        self.nn_r = tf_keras.Sequential([
            tf_keras.Input(shape=(len(self.x_cols), )),
        ])

        for units in list(self.units_for_layers):
            self.nn_r.add(tf_keras.layers.Dense(units, activation='relu'))
        self.nn_r.add(tf_keras.layers.Dense(2*len(self.y_cols), activation='linear'))

        self.nn_r.compile(optimizer=tf_keras.optimizers.Adam(),
                      loss=self.negative_log_likelihood,
                      metrics=[self.mae, 'accuracy'])
    
    @logger_decorator(logger)
    def train(self):
        '''Trains the model.'''
        es = tf_keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.002,
                           patience=10, start_from_epoch=190)
        mc = tf_keras.callbacks.ModelCheckpoint(self.model_path,
                             monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        
        if not self.lr:
            callbacks = [self.custom_callback(self)]
        else:
            call_lr = tf_keras.callbacks.LearningRateScheduler(self.scheduler)
            callbacks = [self.custom_callback(self)]

        if self.with_generator:
            history = self.nn_r.fit(self.df_data, epochs=self.epochs, batch_size=32, validation_split=0.3)
        else:
            history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs, validation_split=0.3,
                      callbacks=callbacks)
        

        # pred_train = nn_r.predict(self.X_train)
        # pred_test = nn_r.predict(self.X_test)
        # idx = 0
        # text = ''
        # for col in self.y_cols:
        #     mae_tr = MAE(self.y_train.iloc[:, idx], pred_train[:, idx])
        #     self.mae_tr_list.append(mae_tr)
        #     mae_te = MAE(self.y_test.iloc[:, idx], pred_test[:, idx])
        #     diff_i = (self.y_test.iloc[:, idx] - pred_test[:, idx])
        #     mean_diff_i = (diff_i).mean()
        #     meae_tr = MeAE(self.y_train.iloc[:, idx], pred_train[:, idx])
        #     meae_te = MeAE(self.y_test.iloc[:, idx], pred_test[:, idx])
        #     median_diff_i = (diff_i).median()
        #     text += f"MAE_train_{col} : {mae_tr:0.5f}\t" + \
        #             f"MAE_test_{col} : {mae_te:0.5f}\t" + \
        #             f"mean_diff_test_pred_{col} : {mean_diff_i:0.5f}\t" + \
        #             f"MeAE_train_{col} {meae_tr:0.5f}\t" + \
        #             f"MeAE_test_{col} {meae_te:0.5f}\t" + \
        #             f"median_diff_test_pred_{col} {median_diff_i:0.5f}\n"
        #     idx = idx + 1

        # nn_r.save(self.model_path)
        with open(os.path.join(os.path.dirname(self.model_path), 'params.txt'), "w") as params_file:
            for key, value in self.params.items():
                params_file.write(f'{key} : {value}\n')
        # with open(os.path.join(os.path.dirname(self.model_path), 'performance.txt'), "w") as text_file:
        #     text_file.write(text)
        # self.text = text
        return history
    
    @logger_decorator(logger)
    def predict(self, start = 0, end = -1, mask_column='index', write_bkg=True, write_frg=False, batch_size=1, save_predictions_plot=False, support_variables=[]) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''Predicts the output data.
        
        Parameters:
        ----------
            start (int): The starting index. Default is 0.
            end (int): The ending index. Defualt is -1.'''
        df_data = Data.get_masked_dataframe(data=self.df_data, start=start, stop=end, column=mask_column, reset_index=False)
        if df_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        scaled_data = self.scaler.transform(df_data[self.x_cols])
        if batch_size > 1:
            y_pred = np.array([])
            for i in range(0, len(scaled_data), batch_size):
                y_pred = np.append(y_pred, self.nn_r.predict(scaled_data[i:i + batch_size]))
        else:
            y_pred = self.nn_r.predict(scaled_data)
        mean_pred = y_pred[:, :len(self.y_cols)]
        log_var_pred = y_pred[:, len(self.y_cols):]
        std_pred = np.sqrt(np.exp(log_var_pred))
        y_pred = pd.DataFrame(mean_pred, columns=self.y_cols)
        y_std = pd.DataFrame(std_pred, columns=[f'{col}_std' for col in self.y_cols])
        y_pred = pd.concat([y_pred, y_std], axis=1)
        y_pred['datetime'] = df_data['datetime'].values
        y_pred.reset_index(drop=True, inplace=True)
        df_ori = df_data[self.y_cols].reset_index(drop=True)
        df_ori['datetime'] = df_data['datetime'].values
        df_ori.reset_index(drop=True, inplace=True)
        if write_bkg:
            path = os.path.join(os.path.dirname(self.model_path))
            if not self.model_id:
                path = os.path.dirname(self.model_path)
            File.write_df_on_file(y_pred, os.path.join(path, 'bkg'))
            gc.collect()

            if write_frg:
                path = os.path.join(os.path.dirname(self.model_path))
                if not self.model_id:
                    path = os.path.dirname(self.model_path)
                File.write_df_on_file(df_ori, os.path.join(path, 'frg'))
        if save_predictions_plot:
            y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(self.y_pred_cols, self.y_cols)}).drop(columns=self.y_cols)
            tiles_df = Data.merge_dfs(df_data[self.y_cols_raw + ['datetime'] + support_variables], y_pred)
            self.save_predictions_plots(tiles_df, start, end, self.params)
        return df_ori, y_pred

class RNNPredictor(FFNNPredictor):
    logger = Logger('RNN').get_logger()

    def __init__(self, df_data, y_cols, x_cols, y_cols_raw, y_pred_cols):
        super().__init__(df_data, y_cols, x_cols, y_cols_raw, y_pred_cols)

    @logger_decorator(logger)
    def reshape_data(self, x, y):
        '''Reshapes the data for the RNN model.'''
        y = y[self.params['timesteps']:]
        x = np.array([x[i:i + self.params['timesteps']] for i in np.arange(len(x) - self.params['timesteps'])])
        x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))
        return x, y

    @logger_decorator(logger)
    def create_model(self):
        '''Creates the model.'''
        
        inputs = Input(shape=(None, len(self.x_cols)))
        hidden = LSTM(90)(inputs)
        for count, units in enumerate(list(self.units_for_layers)):
            hidden = Dense(units, activation='relu')(hidden)
            if self.norm:
                hidden = BatchNormalization()(hidden)
            if self.drop:
                hidden = Dropout(self.do)(hidden)
        outputs = Dense(len(self.y_cols), activation='linear')(hidden)

        self.nn_r = Model(inputs=[inputs], outputs=outputs)
        plot_model(self.nn_r, to_file=os.path.join(os.path.dirname(self.model_path), 'schema.png'),
                   show_shapes=True, show_layer_names=True, rankdir='TB')

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
    def train(self):
        '''Trains the model.'''
        self.X_train, self.y_train = self.reshape_data(self.X_train, self.y_train)
        self.X_test, self.y_test = self.reshape_data(self.X_test, self.y_test)
        
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
        with open(os.path.join(os.path.dirname(self.model_path), 'params.txt'), "w") as params_file:
            for key, value in self.params.items():
                params_file.write(f'{key} : {value}\n')
        with open(os.path.join(os.path.dirname(self.model_path), 'performance.txt'), "w") as text_file:
            text_file.write(text)
        self.text = text
        return history

    @logger_decorator(logger)
    def predict(self, start = 0, end = -1, write_bkg=True, write_frg=False, batch_size=1, save_predictions_plot=False, support_variables=[]) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''Predicts the output data.
        
        Parameters:
        ----------
            start (int): The starting index. (Default is 0).
            end (int): The ending index. (Defualt is -1).
            write (bool): If the predicted and original dataset will be written in a file. (Defualt is True)
            batched (bool): If the dataset will be modeled in batch. (Defualt is False)'''
        df_data = Data.get_masked_dataframe(data=self.df_data, start=start, stop=end, reset_index=False)
        if df_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        data = self.scaler.transform(df_data)
        data = np.array([data[i:i + self.params['timesteps']] for i in np.arange(len(data) - self.params['timesteps'])])
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))
        if batch_size > 1:
            pred_x_tot = np.array([])
            batch_size = len(data)//self.params['timesteps']
            for i in range(0, len(data), batch_size):
                pred_x_tot = np.append(pred_x_tot, self.nn_r.predict(data[i:i + batch_size]))
            pred_x_tot = np.reshape(pred_x_tot, (len(data), len(self.y_cols)))
        else:
            pred_x_tot = self.nn_r.predict(data)
        gc.collect()

        df_ori = self.df_data[start:end][self.y_cols].reset_index(drop=True)
        y_pred = pd.DataFrame(pred_x_tot, columns=self.y_cols)
        df_ori['datetime'] = self.df_data[start:end]['datetime'].values
        y_pred['datetime'] = self.df_data[start+self.timesteps:end]['datetime'].values

        df_ori.reset_index(drop=True, inplace=True)
        y_pred.reset_index(drop=True, inplace=True)
        if write_bkg:
            path = os.path.join(os.path.dirname(self.model_path))
            if not self.model_id:
                path = os.path.dirname(self.model_path)
            File.write_df_on_file(y_pred, os.path.join(path, 'bkg'))
            gc.collect()

            if write_frg:
                path = os.path.join(os.path.dirname(self.model_path))
                if not self.model_id:
                    path = os.path.dirname(self.model_path)
                File.write_df_on_file(df_ori, os.path.join(path, 'frg'))
        if save_predictions_plot:
            y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(self.y_pred_cols, self.y_cols)}).drop(columns=self.y_cols)
            tiles_df = Data.merge_dfs(df_data[self.y_cols_raw + ['datetime'] + support_variables], y_pred)
            self.save_predictions_plots(tiles_df, start, end, self.params)
        return df_ori, y_pred
        

class MedianKNeighborsRegressor(KNeighborsRegressor):
    '''The class for the Median K-Nearest Neighbors model.'''
    logger = Logger('MedianKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
    def predict(self, X):
        '''Predict the target for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
            Target values.
        '''
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