import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

@tf_keras.utils.register_keras_serializable()
class CustomLosses:
    def __init__(self, normalizations_dict):
        '''The class for custom loss functions.'''
        self.normalizations_dict = normalizations_dict

    def get_config(self):
        '''Returns the configuration of the custom loss functions.'''
        pass

    @tf_keras.utils.register_keras_serializable()
    def mae(self, y_true, y_pred):
        '''The mean absolute error metric computed using the mean of the output.'''
        return tf.reduce_mean(tf.abs(y_true - y_pred)) / self.normalizations_dict['mae']

    @tf_keras.utils.register_keras_serializable()
    def mae_bnn(self, y_true, y_pred):
        '''The mean absolute error metric computed using the mean of the output.'''
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        return tf.reduce_mean(tf.abs(y_true - mean)) / self.normalizations_dict['mae']

    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood_stddev(self, y_true, y_pred):
        '''The negative log likelihood loss function computed using the mean and standard deviation.'''
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        var = y_pred[:, num_outputs:]
        norm_dist = tfd.Normal(loc=mean, scale=1e-3 + tf.math.softplus(0.05*var))
        nll = -norm_dist.log_prob(y_true)
        return nll

    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood(self, y_true, y_pred):
        '''The negative log likelihood loss function computed using the mean and variance.'''
        mean = y_pred.mean()
        log_var = y_pred.variance()
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        return tf.reduce_mean(nll)
    
    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood_bnn(self, y_true, y_pred):
        '''The negative log likelihood loss function computed using the mean and variance.'''
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        stddev = y_pred[:, num_outputs:]
        nll = 0.5 * tf.square((y_true - mean) / stddev) + tf.math.log(stddev)
        
        return tf.reduce_mean(nll)

    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood_var(self, y_true, y_pred):
        '''The negative log likelihood loss function computed using the mean and variance.'''
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        return tf.reduce_mean(nll)
    
    @tf_keras.utils.register_keras_serializable()
    def aic(self, y_true, y_pred):
        '''The Akaike Information Criterion loss function.'''
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        return tf.reduce_mean(nll) + 2 * num_outputs

    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood_huber(self, y_true, y_pred, delta=0.003):
        '''Gaussian negative log-likelihood with a Huber loss for robustness.'''
        num_outputs = tf.shape(y_true)[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # huber loss
        residual = tf.abs(y_true - mean)
        huber_loss = tf.where(residual <= delta, 0.5 * tf.square(residual), delta * (residual - 0.5 * delta))
        nll = 0.5 * log_var + huber_loss * tf.exp(-log_var)
        return tf.reduce_mean(nll)

    @tf_keras.utils.register_keras_serializable()
    def mse(self, y_true, y_pred):
        '''The mean squared error metric computed using the mean of the output.'''
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        return tf.reduce_mean(tf.square(y_true - mean)) / self.normalizations_dict['mse']

    @tf_keras.utils.register_keras_serializable()
    def r_squared(self, y_true, y_pred):
        '''The R-squared metric computed using the mean of the output.'''
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        ss_res = tf.reduce_sum(tf.square(y_true - mean))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0)))
        return 1 - ss_res / (ss_tot + tf_keras.backend.epsilon())

    @tf_keras.utils.register_keras_serializable()
    def NLL(self, y_true, y_pred):
        '''The negative log likelihood loss function computed using the mean and variance.'''
        return -y_pred.log_prob(y_true)

    @tf_keras.utils.register_keras_serializable()
    def nll_metric(self, y_true, y_pred):
        '''The negative log likelihood loss function computed using the mean and variance.'''
        mean = tf.reduce_mean(y_pred)
        log_var = tf.math.log(tf.math.reduce_variance(y_pred))
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        return tf.reduce_mean(nll)

    @tf_keras.utils.register_keras_serializable()
    def NLL_median(self, y_true, y_pred):
        '''The negative log likelihood loss function computed using the mean and variance.'''
        return tfp.stats.percentile(y_pred.log_prob(y_true),50., interpolation='midpoint')

    @tf_keras.utils.register_keras_serializable()
    def MSE(self, y_true, y_pred):
        '''The mean squared error loss function computed using the mean of the output.'''
        return tf.reduce_mean(tf.square(y_true - y_pred.mean()))

    @tf_keras.utils.register_keras_serializable()
    def MAE(self, y_true, y_pred):
        '''The mean absolute error loss function computed using the mean of the output.'''
        return tf.reduce_mean(tf.abs(y_true - y_pred.mean()))

    @tf_keras.utils.register_keras_serializable()
    def KL_divergence(self, posterior, prior):
        '''The Kullback-Leibler divergence loss function.'''
        return tfp.distributions.kl_divergence(posterior, prior)
    
    @tf_keras.utils.register_keras_serializable()
    def spectral_loss(self, y_true, y_pred):
        y_true_fft = tf.signal.fft(tf.cast(y_true, tf.complex64))
        y_pred_fft = tf.signal.fft(tf.cast(y_pred, tf.complex64))
        spectral_loss = tf.reduce_mean(tf.square(tf.abs(y_true_fft - y_pred_fft)))
        mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        return spectral_loss + mae_loss
