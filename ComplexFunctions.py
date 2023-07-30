import tensorflow as tf

import pickle
copy_class = lambda class_obj: pickle.loads(pickle.dumps(class_obj))

#%%
import numpy as np
pi_2_degree = 180./np.pi
'''
    Complex Losses
'''
def ComplexEnergyLoss(y_true: tf.complex64, y_pred: tf.complex64, epsilon: tf.float32=1e-8) -> tf.float32:
    ''' 
        chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2101.12249.pdf
    '''
    
    # J = ([ln(r_pred/r_true)]^2 + (theta_true - theta_pred)^2)/2
    magni_err = LogRatio(y_true, y_pred, epsilon)
    theta_err = PhaseMSE(y_true, y_pred)
    
    return (magni_err + theta_err)/2.

def LogRatio(y_true, y_pred, epsilon=1e-8):
    y_true_magni = tf.math.abs(y_true)
    y_pred_magni = tf.math.abs(y_pred)
    
    # J = (ln(r_pred/r_true)^2
    magni_err = tf.math.square(tf.math.log(y_pred_magni/y_true_magni + epsilon))
    
    return tf.math.reduce_mean(magni_err, axis=-1)

def PhaseMSE(y_true, y_pred):
    '''
        For theta, we have to take periodicity [-pi, pi] into consideration.
        e.g. 1) theta_a=-3 and theta_b=3, abs diff is 2*pi - 3 - 3 = 0.2832 but NOT 3 - (-3) = 6!
             2) theta_a=-3 and theta_b=4, abs diff is 4 + 3 - 2*pi = 0.7168
    '''
    y_true_theta = tf.math.angle(y_true)
    y_pred_theta = tf.math.angle(y_pred)

     # J = (theta_true - theta_pred)^2
    diff_theta = y_pred_theta - y_true_theta
    theta_err = tf.math.square(tf.atan2(tf.sin(diff_theta), tf.cos(diff_theta)) * pi_2_degree)
    
    return tf.math.reduce_mean(theta_err, axis=-1)

def AngleCosLoss(y_true, y_pred):
    y_true_theta = tf.math.angle(y_true)
    y_pred_theta = tf.math.angle(y_pred)

     # J = 2(1 − cos(theta_true - theta_pred)) approx (theta_true - theta_pred)**2 + O((theta_true - theta_pred)**4)
    diff_theta = y_pred_theta - y_true_theta
    theta_err = 2*(1 - tf.math.cos(diff_theta)) * pi_2_degree
    return tf.math.reduce_mean(theta_err, axis=-1)

def ComplexMSE(y_true, y_pred):
    square_err = tf.math.square(tf.math.abs(y_true - y_pred))
    return tf.math.reduce_mean(square_err, axis=-1)

#%%
'''
    Complex Metrics for current iteration ONLY
    Loss = averaging Metrics of all previous iterations
'''
class ComplexEnergyMetric(tf.keras.metrics.Metric):
    def __init__(self, name='complex_energy', epsilon=1e-8, **kwargs):
        super().__init__(name=name, **kwargs)
        self.metric = self.add_weight(name=name, initializer='zeros')
        self.epsilon = epsilon
  
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.reset_state()
        
        magni_err = LogRatio(y_true, y_pred, self.epsilon)
        theta_err = PhaseMSE(y_true, y_pred)
        self.metric.assign_add(tf.math.reduce_mean((magni_err + theta_err))/2.)

    def result(self):
        return self.metric
    
#%%
class BaseMetric(tf.keras.metrics.Metric):
    def __init__(self, name, loss_func, **kwargs):
        super().__init__(name=name, **kwargs)
        self.metric = self.add_weight(name=name, initializer='zeros')
        self.loss_func = loss_func
  
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.reset_state()
        self.metric.assign_add(tf.math.reduce_mean(self.loss_func(y_true, y_pred)))

    def result(self):
        return self.metric

class LogRatioMetric(BaseMetric):
    def __init__(self, name='log_ratio', epsilon=1e-8, **kwargs):
        loss_func = lambda x, y: LogRatio(x, y, epsilon)
        super().__init__(name=name, loss_func=loss_func, **kwargs)

class PhaseMSEMetric(BaseMetric):
    def __init__(self, name='phase_mse', **kwargs):
        super().__init__(name=name, loss_func=PhaseMSE, **kwargs)
    
class AngleCosMetric(BaseMetric):
    def __init__(self, name='angle_cos', **kwargs):
        super().__init__(name=name, loss_func=AngleCosLoss, **kwargs)
    
class ComplexMSEMetric(BaseMetric):
    def __init__(self, name='complex_mse', **kwargs):
        super().__init__(name=name, loss_func=ComplexMSE, **kwargs)

#%%
'''
    Complex Activation Functions
'''
def ComplexCardioid(x):
    scale = 0.5 * (1 + tf.math.cos(tf.math.angle(x)))
    output = tf.complex(tf.math.real(x) * scale, tf.math.imag(x) * scale)
    return output

def ComplexSoftmax(x, axis=None):
    return tf.nn.softmax(tf.abs(x), axis=axis)

def ComplexSquareSoftmax(x):
    return tf.nn.softmax(tf.math.square(tf.abs(x)))

class ComplexActivation(tf.keras.layers.Layer):
    '''
        Phase-amplitude activation : f(z) = f(|r| + b) * e^(iθ)
    '''
    def __init__(self, act_fun="relu",
                 b_initializer="zeros",
                 b_regularizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.act_fun = tf.keras.layers.Activation(act_fun)
        self.b_initializer = tf.keras.initializers.get(b_initializer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)
    
    def build(self, input_shape):
        self.shape = tuple(1 for i in input_shape[:-1]) + (-1,)
        
        self.b = self.add_weight(
            name='b',
            shape=(input_shape[-1],),
            initializer=self.b_initializer,
            regularizer=self.b_regularizer,
            trainable=True
        )
    
    def call(self, inputs):
        magni, theta = tf.math.abs(inputs), tf.math.angle(inputs)
        r = tf.complex(self.act_fun(magni + tf.reshape(self.b, self.shape)), 0.)
        phase = tf.exp(tf.complex(0., theta))
        outputs = r * phase
        
        return  outputs

class GLUActivation(tf.keras.layers.Layer):
    def __init__(self, act_fun="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.act_fun = tf.keras.layers.Activation(act_fun)
        
    def build(self, input_shape):
        self.split_idx = input_shape[-1] // 2
    
    def call(self, inputs):
        a, b = inputs[...,:self.split_idx], inputs[...,self.split_idx:]
        outputs = a * self.act_fun(b)
        
        return outputs
    
class ComplexGLUActivation(tf.keras.layers.Layer):
    def __init__(self, act_fun="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.act_fun = tf.keras.layers.Activation(act_fun)
        
    def build(self, input_shape):
        self.split_idx = input_shape[-1] // 2
    
    def call(self, inputs):
        a, b = inputs[...,:self.split_idx], inputs[...,self.split_idx:]
        
        magni, theta = tf.math.abs(b), tf.math.angle(b)
        r = tf.complex(self.act_fun(magni), 0.)
        phase = tf.exp(tf.complex(0., theta))
        outputs = a * r * phase
        
        return outputs

class DoubleActivation(tf.keras.layers.Layer):
    def __init__(self, act_fun_1="tanh", act_fun_2="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.act_fun_1 = tf.keras.layers.Activation(act_fun_1)
        self.act_fun_2 = tf.keras.layers.Activation(act_fun_2)
    
    def call(self, inputs):
        outputs = self.act_fun_1(inputs) * self.act_fun_2(inputs)
        
        return outputs

class DoubleComplexActivation(tf.keras.layers.Layer):
    def __init__(self, 
                 act_fun_1="tanh", 
                 act_fun_2="sigmoid", 
                 b_initializer="zeros",
                 b_regularizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.act_fun_1 = tf.keras.layers.Activation(act_fun_1)
        self.act_fun_2 = tf.keras.layers.Activation(act_fun_2)
        
        self.b_initializer = tf.keras.initializers.get(b_initializer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)
    
    def build(self, input_shape):
        self.shape = tuple(1 for i in input_shape[:-1]) + (-1,)
        
        self.b_1 = self.add_weight(
            name='b1',
            shape=(input_shape[-1],),
            initializer=copy_class(self.b_initializer),
            regularizer=self.b_regularizer,
            trainable=True
        )
        
        self.b_2 = self.add_weight(
            name='b2',
            shape=(input_shape[-1],),
            initializer=copy_class(self.b_initializer),
            regularizer=self.b_regularizer,
            trainable=True
        )
    
    def call(self, inputs):
        magni, theta = tf.math.abs(inputs), tf.math.angle(inputs)
        r_1 = self.act_fun_1(magni + tf.reshape(self.b_1, self.shape))
        r_2 = self.act_fun_2(magni + tf.reshape(self.b_2, self.shape))
        phase = tf.exp(tf.complex(0., theta))
        # take absolute such that phase remains unchanged
        outputs = tf.complex(tf.abs(r_1 * r_2), 0.) * phase
        
        return outputs

#%%
'''
    Weight Initialization
'''
def _compute_fans(shape):
    if len(shape) < 1:                  # for unit scalar
        fan_in = fan_out = 1.
    elif len(shape) == 1:               # BatchNormalization / LayerNormalization(axis=-1)
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:               # Dense / RNN : (channels, filters)
        fan_in, fan_out = shape
    else:
        # CNN : (kernel_height, kernel_width, channels, filters)
        filter_size = 1
        for dim in shape[:-2]:
            filter_size *= dim
        fan_in = shape[-2] * filter_size
        fan_out = shape[-1] * filter_size
    return fan_in, fan_out

def Demucs_Rescaling(x, a=0.1):
    std = tf.math.reduce_std(x, axis=None, keepdims=True)
    alpha = std / a
    return x / tf.sqrt(alpha)

#%%
'''
    {z in Complex U(0, 1): |z| <= 1}
    Complex N(0, 1) = Real N(0, 1/2) + Imaginary N(0, 1/2)
    Uniform : S-type activation function
    Normal  : ReLU-type activation function
'''

class GlorotUniform():
    def __init__(self, seed=None):
        self.seed = seed
    
    def __call__(self, shape, dtype=None, **kwargs):
        r = tf.math.sqrt(6 / sum(_compute_fans(shape)))
        return tf.random.uniform(
            shape, minval=-r, maxval=r, seed=self.seed, dtype=dtype
        )

class GlorotNormal():
    def __init__(self, seed=None):
        self.seed = seed
    
    def __call__(self, shape, dtype=None, **kwargs):
        std = tf.math.sqrt(2 / sum(_compute_fans(shape)))
        return tf.random.normal(
            shape, mean=0, stddev=std, seed=self.seed, dtype=dtype
        )

class HeUniform():
    def __init__(self, seed=None):
        self.seed = seed
    
    def __call__(self, shape, dtype=None, **kwargs):
        fan_in, fan_out = sum(_compute_fans(shape))
        r = tf.math.sqrt(1 / fan_in)
        return tf.random.uniform(
            shape, minval=-r, maxval=r, seed=self.seed, dtype=dtype
        )

class HeNormal():
    def __init__(self, seed=None):
        self.seed = seed
    
    def __call__(self, shape, dtype=None, **kwargs):
        fan_in, fan_out = sum(_compute_fans(shape))
        std = tf.math.sqrt(2 / fan_in)
        return tf.random.normal(
            shape, mean=0, stddev=std, seed=self.seed, dtype=dtype
        )