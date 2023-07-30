import tensorflow as tf
from cvnn.layers import ComplexDense

from ComplexFunctions import ComplexSoftmax, ComplexCardioid

import numpy as np
import pickle
copy_class = lambda class_obj: pickle.loads(pickle.dumps(class_obj))

#%%
def magtheta_2_spec(inputs):
    magni, theta = inputs
    return tf.complex(magni, 0.) * tf.exp(tf.complex(0., theta))

def spec_2_magtheta(inputs):
    return tf.abs(inputs), tf.math.angle(inputs)

def magphase_2_spec(inputs):
    magni, phase = inputs
    return tf.complex(magni, 0.) * phase

def spec_2_magphase(inputs):
    magni, theta = spec_2_magtheta(inputs)
    return magni, tf.exp(tf.complex(0., theta))

def spec_2_phase(inputs):
    theta = tf.math.angle(inputs)
    return tf.exp(tf.complex(0., theta))
    
#%%
class PositionalEncoding1D(tf.keras.layers.Layer):
    '''
        PE(tau,2i)   = sin(tau/n^(2i/D_Model))
        PE(tau,2i+1) = cos(tau/n^(2i/D_Model))
        outputs = inputs + PE
    '''
    def __init__(self, n=10_000, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        
    def build(self, input_shape):
        _, T, n_channels = input_shape
        channels = int(tf.math.ceil(n_channels/2) * 2)
        
        tau = tf.range(T, dtype=self.dtype)
        inv = 1./tf.math.pow(self.n, tf.range(0, channels, 2)/channels)
        tau_inv = tf.einsum("i,j->ij", tau, tf.cast(inv, self.dtype))
        
        PE = tf.stack((tf.sin(tau_inv), tf.cos(tau_inv)), -1)
        self.EncodingMatrix = tf.reshape(PE, (1, T, channels))[...,:n_channels]

    @tf.function
    def call(self, inputs):
        return inputs + self.EncodingMatrix
    
class PositionalEncoding2D(tf.keras.layers.Layer):
    '''
        Paper: Translating Math Formula Images to LaTeX Sequences Using Deep Neural Networks with Sequence-level Training
        PE(tau_x,tau_y,2i)             = sin(tau_x/n^(4i/D_Model))
        PE(tau_x,tau_y,2i+1)           = cos(tau_x/n^(4i/D_Model))
        PE(tau_x,tau_y,2j+D_Model/2)   = sin(tau_y/n^(4j/D_Model))
        PE(tau_x,tau_y,2j+1+D_Model/2) = cos(tau_y/n^(4j/D_Model))
        outputs = inputs + PE
    '''
    def __init__(self, n=10_000, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        
    def build(self, input_shape):
        _, T_x, T_y, n_channels = input_shape
        channels = int(tf.math.ceil(n_channels/4) * 4)

        T_max = max(T_x, T_y)
        tau_max = tf.range(T_max, dtype=self.dtype)
        inv = 1./tf.math.pow(self.n, tf.range(0, channels, 4)/channels)
        tau_max_inv = tf.einsum("i,j->ij", tau_max, tf.cast(inv, self.dtype))
        PE_max = tf.stack((tf.sin(tau_max_inv), tf.cos(tau_max_inv)), -1)
        
        PE_x = tf.reshape(PE_max, (1, T_max, 1, channels//2))[:,:T_x,...]
        PE_y = tf.reshape(PE_max, (1, 1, T_max, channels//2))[...,:T_y,:]
        PE_x = tf.tile(PE_x, (1, 1, T_y, 1))
        PE_y = tf.tile(PE_y, (1, T_x, 1, 1))
        self.EncodingMatrix = tf.concat((PE_x, PE_y), -1)[...,:n_channels]

    @tf.function
    def call(self, inputs):
        return inputs + self.EncodingMatrix
     
#%%
class ComplexEmbedding(tf.keras.layers.Layer):
    def __init__(self, use_theta=False, **kwargs):
        super().__init__(**kwargs)
        self.use_theta = use_theta
        
    def build(self, input_shape):
        _, *dims = input_shape
        
        self.r = tf.Variable(tf.random.uniform(dims, 0, 1),
                             name="r" ,
                             dtype=self.dtype,
                             constraint=lambda r: tf.clip_by_value(r, 0, 1),
                             trainable=True)
        
        self.w = tf.Variable(tf.random.uniform(dims, 0, 2*np.pi),
                             name="w",
                             dtype=self.dtype,
                             trainable=True)
        
        if self.use_theta:
            self.theta = tf.Variable(tf.random.uniform(dims, 0, 2*np.pi),
                                     name="theta",
                                     dtype=self.dtype,
                                     trainable=True)
        else:
            self.theta = 0.
    
    @tf.function
    def call(self, inputs):
        PE = tf.complex(self.r, 0.) * tf.exp(tf.complex(0., self.tau * self.w + self.theta))
        return inputs + tf.expand_dims(PE, axis=0)


class ComplexEmbedding1D(ComplexEmbedding):
    '''
        Paper : Encoding word order in complex embeddings
        PE(tau,j) = r_j exp[i(tau x w_j + θ_j)] with |r_j| < 1
        outputs = inputs + PE
    '''
    def __init__(self, use_theta=False, **kwargs):
        super().__init__(use_theta, **kwargs)
        
    def build(self, input_shape):
        super().build(input_shape)
        
        _, T, n_channels = input_shape
        self.tau = tf.expand_dims(tf.range(T, dtype=self.dtype), axis=-1)

class ComplexEmbedding2D(ComplexEmbedding):
    '''
        PE(tau_x, tau_y, j) = r_j exp[i(sqrt(tau_x^2 + tau_y^2) x w_j + θ_j)] with |r_j| < 1
        outputs = inputs + PE
    '''
    def __init__(self, use_theta=False, **kwargs):
        super().__init__(use_theta, **kwargs)
        
    def build(self, input_shape):
        super().build(input_shape)
        
        _, T_x, T_y, n_channels = input_shape
        tau_x = tf.reshape(tf.range(T_x, dtype=self.dtype), shape=(1, -1, 1))
        tau_y = tf.reshape(tf.range(T_y, dtype=self.dtype), shape=(1, 1, -1))
        self.tau = tf.expand_dims(tf.sqrt(tf.math.square(tau_x) + tf.math.square(tau_y)), axis=-1)

#%%
class BaseGLU(tf.keras.layers.Layer):
    '''
        Gated Linear Unit : GLU Variants Improve Transformer
        outputs = act_fun(inputs W + b) x (inputs V + c)
        For inputs with more than 2 dimensions, first flatten, then dense, finally reshape back
    '''
    def __init__(self, units=None, act_fun="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.act_fun  = tf.keras.layers.Layer.Activation(act_fun)
    
    def build(self, input_shape):
        if self.units is None:
            self.units = input_shape[-1]
    
    @tf.function
    def call(self, inputs):
        x = self.linear(inputs)
        outputs = x[...,:self.units] * self.act_fun(x[...,self.units:])
        
        return outputs

class GLU(BaseGLU):
    def __init__(self, units=None, act_fun="sigmoid", **kwargs):
        super().__init__(units, act_fun, **kwargs)
    
    def build(self, input_shape):
        super().build(input_shape)
        self.linear = tf.keras.layers.Layer.Dense(self.units*2)
    
class ComplexGLU(BaseGLU):
    def __init__(self, units=None, act_fun=ComplexCardioid, **kwargs):
        super().__init__(units, act_fun, **kwargs)
    
    def build(self, input_shape):
        super().build(input_shape)
        self.linear = ComplexDense(self.units*2)


#%%
class BaseAttention(tf.keras.layers.Layer):
    ''' Base class for Single/Multi-head attention
        If use_scale, we compute alpha x score with alpha being a scalar;
        If use_causal_mask, it creates a lower triangular matrix from scores and 
        prevents the flow of future information to the past;
    
    '''
    def __init__(self, key_dim, value_dim=None, 
                 use_scale=False, use_causal_mask=False,
                 kernel_initializer="glorot_uniform",
                 alpha_initializer="glorot_normal",
                 kernel_regularizer=None,
                 alpha_regularizer=None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.key_dim = key_dim
        self.value_dim = value_dim or key_dim
        self.use_scale = use_scale
        self.use_causal_mask = use_causal_mask
        
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
        
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.alpha_regularizer = tf.keras.regularizers.get(alpha_regularizer)
    
    def build_weight(self, K_shape, add_shape=(), add_name=""):
        *_, T, D_Model = K_shape
        W_Q = self.add_weight(
            name=f'query weight{add_name}',
            shape=add_shape + (D_Model, self.key_dim),
            initializer=copy_class(self.kernel_initializer),
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        
        W_V = self.add_weight(
            name=f'value weight{add_name}',
            shape=add_shape + (D_Model, self.value_dim),
            initializer=copy_class(self.kernel_initializer),
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        
        W_K = self.add_weight(
            name=f'key weight{add_name}',
            shape=add_shape + (D_Model, self.key_dim),
            initializer=copy_class(self.kernel_initializer),
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        
        alpha = 1.
        if self.use_scale:
            alpha = self.add_weight(
                name=f'alpha{add_name}',
                shape=add_shape + (T, 1),
                initializer=copy_class(self.alpha_initializer),
                regularizer=self.alpha_regularizer,
                trainable=True
            )
            
        return W_Q, W_V, W_K, alpha
    
    def _get_attention(self, query, value, key):
        query_W = tf.matmul(query, self.W_Q)
        value_W = tf.matmul(value, self.W_V)
        key_W = tf.matmul(key, self.W_K)
        
        scores = tf.matmul(query_W, tf.transpose(key_W, perm=self.perm))/self.sqrt_D
        if self.use_scale:
            scores *= self.alpha
        
        if self.use_causal_mask:
            scores += self.causal_mask
        
        head = tf.matmul(self.softmax(scores), value_W)
        return head
    
    def _get_relative_attention(self, query, value, key):
        query_W = tf.matmul(query, self.W_Q)
        value_W = tf.matmul(value, self.W_V)
        key_W = tf.matmul(key, self.W_K)
        
        embedding_Q = self.relative_Q._get_embedding()
        relative_Q_W = tf.matmul(tf.expand_dims(query_W, axis=-2), tf.transpose(embedding_Q, perm=[0, 2, 1])) /self.sqrt_D
            
        scores = tf.matmul(query_W, tf.transpose(key_W, perm=self.perm))/self.sqrt_D + tf.squeeze(relative_Q_W, axis=-2)
        
        if self.use_scale:
            scores *= self.alpha
        
        if self.use_causal_mask:
            scores += self.causal_mask
        
        softmax_scores = self.softmax(scores)
        embedding_S = self.relative_S._get_embedding()
        relative_Smax = tf.matmul(tf.expand_dims(softmax_scores, axis=-2), embedding_S)

        heads = tf.matmul(softmax_scores, value_W) + tf.squeeze(relative_Smax, axis=-2)
        return heads

class Attention(BaseAttention):
    def __init__(self, key_dim, **kwargs):
        super().__init__(key_dim, **kwargs)
        
        self.sqrt_D = tf.sqrt(float(key_dim))
        self.softmax = tf.nn.softmax
        self.perm = [0, 2, 1]
        
    def build(self, input_shape):
        # input_shape: [(batch_size, T, D_Model_Q), (batch_size, T, D_Model_V)] or 
        # [(batch_size, T, D_Model_Q), (batch_size, T, D_Model_V), (batch_size, T, D_Model_K)]
        if len(input_shape) == 2:
            K_shape, V_shape = input_shape
        else:
            Q_shape, V_shape, K_shape = input_shape
        
        self.W_Q, self.W_V, self.W_K, self.alpha = self.build_weight(K_shape)
        if self.use_causal_mask:
            T = K_shape[1]
            lower_tri = tf.linalg.LinearOperatorLowerTriangular(tf.ones((1, T, T))).to_dense()
            self.causal_mask = -1e9 * tf.cast(tf.less(lower_tri, 0.5), tf.float32)

    @tf.function
    def call(self, inputs):
        '''
        Parameters
        ----------
        inputs : list of tensors [X_Q, X_V] or [X_Q, X_V, X_K]
        X_Q : (Batch_size, T, D_Model_Q)
        X_V : (Batch_size, T, D_Model_V)
        X_K : (Batch_size, T, D_Model_K)

        X_Q W_Q : Query tensor of shape (Batch_size, T, D_Q)
        X_V W_V : Value tensor of shape (Batch_size, T, D_V)
        X_K W_K : Key tensor of shape (Batch_size, T, D_K)
        '''
        if len(inputs) == 2:
            query, value = inputs
            key = value
        else:
            query, value, key = inputs
        outputs = self._get_attention(query, value, key)
        return outputs
    
class ComplexAttention(BaseAttention):
    ''' Single head attention
        If use_scale, we compute alpha x score with alpha being a scalar
        
        If use_causal_mask, it creates a lower triangular matrix from scores and 
        prevents the flow of future information to the past
    
    '''
    def __init__(self, key_dim, **kwargs):
        super().__init__(key_dim, **kwargs)
        
        self.sqrt_D = tf.cast(tf.sqrt(float(key_dim)), dtype=tf.complex64)
        self.softmax = lambda x: tf.complex(ComplexSoftmax(x), 0.)
        self.perm = [0, 2, 1]
        
    def build(self, input_shape):
        # input_shape: [(batch_size, T, D_Model_Q), (batch_size, T, D_Model_V)] or 
        # [(batch_size, T, D_Model_Q), (batch_size, T, D_Model_V), (batch_size, T, D_Model_K)]
        if len(input_shape) == 2:
            K_shape, V_shape = input_shape
        else:
            Q_shape, V_shape, K_shape = input_shape
        
        self.W_Q_r, self.W_V_r, self.W_K_r, self.alpha_r = self.build_weight(K_shape, add_name=" real")
        self.W_Q_i, self.W_V_i, self.W_K_i, self.alpha_i = self.build_weight(K_shape, add_name=" imag")
        
        if self.use_causal_mask:
            T = K_shape[1]
            lower_tri = tf.linalg.LinearOperatorLowerTriangular(tf.ones((1, T, T))).to_dense()
            causal_mask = -1e9 * tf.cast(tf.less(lower_tri, 0.5), tf.float32)
            self.causal_mask = tf.complex(causal_mask, causal_mask)

    @tf.function
    def call(self, inputs):
        '''
        Parameters
        ----------
        inputs : list of tensors [X_Q, X_V] or [X_Q, X_V, X_K]
        X_Q : (Batch_size, T, D_Model_Q)
        X_V : (Batch_size, T, D_Model_V)
        X_K : (Batch_size, T, D_Model_K)

        X_Q W_Q : Query tensor of shape (Batch_size, T, D_Q)
        X_V W_V : Value tensor of shape (Batch_size, T, D_V)
        X_K W_K : Key tensor of shape (Batch_size, T, D_K)
        '''
        if len(inputs) == 2:
            query, value = inputs
            key = value
        else:
            query, value, key = inputs
            
        self.W_Q = tf.complex(self.W_Q_r, self.W_Q_i)
        self.W_V = tf.complex(self.W_V_r, self.W_V_i)
        self.W_K = tf.complex(self.W_K_r, self.W_K_i)
        self.alpha = tf.complex(self.alpha_r, self.alpha_i)
        
        outputs = self._get_attention(query, value, key)
        return outputs

#%% 
class MultiHeadAttention(BaseAttention):
    def __init__(self, 
                 n_heads,
                 key_dim, 
                 **kwargs
                 ):
        super().__init__(key_dim, **kwargs)
        
        self.n_heads = n_heads
        self.sqrt_D = tf.cast(tf.sqrt(float(key_dim)), dtype=tf.float32)
        self.softmax = tf.nn.softmax
        self.perm = [0, 1, 3, 2]
    
    def build(self, input_shape):
        # input_shape: [(batch_size, T, D_Model_Q), (batch_size, T, D_Model_V)] or 
        # [(batch_size, T, D_Model_Q), (batch_size, T, D_Model_V), (batch_size, T, D_Model_K)]
        if len(input_shape) == 2:
            K_shape, V_shape = input_shape
        else:
            Q_shape, V_shape, K_shape = input_shape
            
        _, self.T, D_Model = K_shape
        self.W_Q, self.W_V, self.W_K, self.alpha = self.build_weight(K_shape, add_shape=(self.n_heads,))
        
        self.W = self.add_weight(
            name='head weight',
            shape=(self.n_heads, self.value_dim, D_Model),
            initializer=copy_class(self.kernel_initializer),
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        
        if self.use_causal_mask:
            lower_tri = tf.linalg.LinearOperatorLowerTriangular(tf.ones((1, 1, self.T, self.T))).to_dense()
            self.causal_mask = -1e9 * tf.cast(tf.less(lower_tri, 0.5), tf.float32)
        
    @tf.function
    def call(self, inputs):
        '''
        Parameters
        ----------
        inputs : list of tensors [X_Q, X_V] or [X_Q, X_V, X_K]
        X_Q : (Batch_size, T, D_Model_Q)
        X_V : (Batch_size, T, D_Model_V)
        X_K : (Batch_size, T, D_Model_K)

        X_Q W_Q : Query tensor of shape (Batch_size, T, D_Q)
        X_V W_V : Value tensor of shape (Batch_size, T, D_V)
        X_K W_K : Key tensor of shape (Batch_size, T, D_K)
        '''
        if len(inputs) == 2:
            query, value = inputs
            key = value
        else:
            query, value, key = inputs
        
        query = tf.expand_dims(query, axis=1)
        value = tf.expand_dims(value, axis=1)
        key   = tf.expand_dims(key, axis=1)
        heads = self._get_attention(query, value, key)
        outputs = tf.reduce_sum(tf.matmul(heads, self.W), axis=1)
        
        return outputs

class ComplexMultiHeadAttention(BaseAttention):
    def __init__(self, 
                 n_heads,
                 key_dim, 
                 **kwargs
                 ):
        super().__init__(key_dim, **kwargs)
        
        self.n_heads = n_heads
        self.sqrt_D = tf.cast(tf.sqrt(float(key_dim)), dtype=tf.complex64)
        self.softmax = lambda x: tf.complex(ComplexSoftmax(x), 0.)
        self.perm = [0, 1, 3, 2]
    
    def build(self, input_shape):
        # input_shape: [(batch_size, T, D_Model_Q), (batch_size, T, D_Model_V)] or 
        # [(batch_size, T, D_Model_Q), (batch_size, T, D_Model_V), (batch_size, T, D_Model_K)]
        if len(input_shape) == 2:
            K_shape, V_shape = input_shape
        else:
            Q_shape, V_shape, K_shape = input_shape
        
        _, self.T, D_Model = K_shape
        self.W_Q_r, self.W_V_r, self.W_K_r, self.alpha_r = self.build_weight(K_shape, add_shape=(self.n_heads,), add_name=" real")
        self.W_Q_i, self.W_V_i, self.W_K_i, self.alpha_i = self.build_weight(K_shape, add_shape=(self.n_heads,), add_name=" imag")
        
        self.W_r = self.add_weight(
            name='heads weight real',
            shape=(self.n_heads, self.value_dim, D_Model),
            initializer=copy_class(self.kernel_initializer),
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        
        self.W_i = self.add_weight(
            name='heads weight imag',
            shape=(self.n_heads, self.value_dim, D_Model),
            initializer=copy_class(self.kernel_initializer),
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        
        if self.use_causal_mask:
            lower_tri = tf.linalg.LinearOperatorLowerTriangular(tf.ones((1, 1, self.T, self.T))).to_dense()
            causal_mask = -1e9 * tf.cast(tf.less(lower_tri, 0.5), tf.float32)
            self.causal_mask = tf.complex(causal_mask, causal_mask)

    @tf.function
    def call(self, inputs):
        '''
        Parameters
        ----------
        inputs : list of tensors [X_Q, X_V] or [X_Q, X_V, X_K]
        X_Q : (Batch_size, T, D_Model_Q)
        X_V : (Batch_size, T, D_Model_V)
        X_K : (Batch_size, T, D_Model_K)

        X_Q W_Q : Query tensor of shape (Batch_size, T, D_Q)
        X_V W_V : Value tensor of shape (Batch_size, T, D_V)
        X_K W_K : Key tensor of shape (Batch_size, T, D_K)
        '''
        if len(inputs) == 2:
            query, value = inputs
            key = value
        else:
            query, value, key = inputs
        
        self.W_Q = tf.complex(self.W_Q_r, self.W_Q_i)
        self.W_V = tf.complex(self.W_V_r, self.W_V_i)
        self.W_K = tf.complex(self.W_K_r, self.W_K_i)
        self.alpha = tf.complex(self.alpha_r, self.alpha_i)
        
        query = tf.expand_dims(query, axis=1)
        value = tf.expand_dims(value, axis=1)
        key   = tf.expand_dims(key, axis=1)
        heads = self._get_attention(query, value, key)
        W = tf.complex(self.W_r, self.W_i)
        outputs = tf.reduce_sum(tf.matmul(heads, W), axis=1)
        
        return outputs

#%%
class BaseRelativePosition(tf.keras.layers.Layer):
    def __init__(self, seq_length, units, clip_val=2, 
                 embeddings_initializer='glorot_uniform',
                 embeddings_regularizer=None):
        super().__init__()
        self.units = units
        self.clip_val = clip_val
        
        self.embedding_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embedding_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
    
        tau_K = tf.range(seq_length, dtype=self.dtype)
        relative_mat = tau_K[None,:] - tau_K[:,None]
        relative_mat = tf.clip_by_value(relative_mat, -self.clip_val, self.clip_val)
        self.relative_mat = tf.cast(relative_mat + self.clip_val, dtype=tf.int32)
    
    def build_embedding(self, add_name=""):
        return self.add_weight(
            name=f'embedding{add_name}',
            shape=(2*self.clip_val + 1, self.units),
            initializer=copy_class(self.embedding_initializer),
            regularizer=self.embedding_regularizer,
            trainable=True
        )

    def _get_embedding(self):
        embedding = tf.gather_nd(self.embedding, tf.reshape(self.relative_mat, (-1, 1)))
        embedding = tf.reshape(embedding, self.relative_mat.shape + (self.units,))
        return embedding
        
class RelativePosition(BaseRelativePosition):
    def __init__(self, seq_length, units, clip_val=2, 
                 embeddings_initializer='glorot_uniform',
                 embeddings_regularizer=None):
        super().__init__(seq_length, units, clip_val=clip_val, 
                         embeddings_initializer=embeddings_initializer, 
                         embeddings_regularizer=embeddings_regularizer)
        
        self.embedding = self.build_embedding()

class ComplexRelativePosition(BaseRelativePosition):
    def __init__(self, seq_length, units, clip_val=2, 
                 embeddings_initializer='glorot_uniform',
                 embeddings_regularizer=None):
        super().__init__(seq_length, units, clip_val=clip_val, 
                         embeddings_initializer=embeddings_initializer, 
                         embeddings_regularizer=embeddings_regularizer)
        
        self.embedding_r = self.build_embedding(add_name=" real")
        self.embedding_i = self.build_embedding(add_name=" imag")


class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(self, 
                 n_heads,
                 key_dim, 
                 clip_val=2,
                 embeddings_initializer='glorot_uniform',
                 embeddings_regularizer=None,
                 **kwargs
                 ):
        super().__init__(n_heads, key_dim, **kwargs)

        self.clip_val = clip_val
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        
    def build(self, input_shape):
        super().build(input_shape)
        self.relative_Q = RelativePosition(self.T, self.key_dim, self.clip_val, 
                                           self.embeddings_initializer, 
                                           self.embeddings_regularizer)
        self.relative_S = RelativePosition(self.T, self.value_dim, self.clip_val, 
                                           self.embeddings_initializer, 
                                           self.embeddings_regularizer)

    @tf.function
    def call(self, inputs):
        '''
        Parameters
        ----------
        inputs : list of tensors [X_Q, X_V] or [X_Q, X_V, X_K]
        X_Q : (Batch_size, T, D_Model_Q)
        X_V : (Batch_size, T, D_Model_V)
        X_K : (Batch_size, T, D_Model_K)

        X_Q W_Q : Query tensor of shape (Batch_size, T, D_Q)
        X_V W_V : Value tensor of shape (Batch_size, T, D_V)
        X_K W_K : Key tensor of shape (Batch_size, T, D_K)
        '''
        if len(inputs) == 2:
            query, value = inputs
            key = value
        else:
            query, value, key = inputs
        
        query = tf.expand_dims(query, axis=1)
        value = tf.expand_dims(value, axis=1)
        key   = tf.expand_dims(key, axis=1)
        heads = self._get_relative_attention(query, value, key)
        outputs = tf.reduce_sum(tf.matmul(heads, self.W), axis=1)
        
        return outputs
    
class ComplexRelativeMultiHeadAttention(ComplexMultiHeadAttention):
    def __init__(self, 
                 n_heads,
                 key_dim, 
                 clip_val=2,
                 embeddings_initializer='glorot_uniform',
                 embeddings_regularizer=None,
                 **kwargs
                 ):
        super().__init__(n_heads, key_dim, **kwargs)
        self.clip_val = clip_val
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer

    def build(self, input_shape):
        super().build(input_shape)
        self.relative_Q = ComplexRelativePosition(self.T, self.key_dim, self.clip_val, 
                                                  self.embeddings_initializer, 
                                                  self.embeddings_regularizer)
        self.relative_S = ComplexRelativePosition(self.T, self.value_dim, self.clip_val, 
                                                  self.embeddings_initializer, 
                                                  self.embeddings_regularizer)

    @tf.function
    def call(self, inputs):
        '''
        Parameters
        ----------
        inputs : list of tensors [X_Q, X_V] or [X_Q, X_V, X_K]
        X_Q : (Batch_size, T, D_Model_Q)
        X_V : (Batch_size, T, D_Model_V)
        X_K : (Batch_size, T, D_Model_K)

        X_Q W_Q : Query tensor of shape (Batch_size, T, D_Q)
        X_V W_V : Value tensor of shape (Batch_size, T, D_V)
        X_K W_K : Key tensor of shape (Batch_size, T, D_K)
        '''
        if len(inputs) == 2:
            query, value = inputs
            key = value
        else:
            query, value, key = inputs
        
        self.W_Q = tf.complex(self.W_Q_r, self.W_Q_i)
        self.W_V = tf.complex(self.W_V_r, self.W_V_i)
        self.W_K = tf.complex(self.W_K_r, self.W_K_i)
        self.alpha = tf.complex(self.alpha_r, self.alpha_i)
        self.relative_Q.embedding = tf.complex(self.relative_Q.embedding_r, self.relative_Q.embedding_i)
        self.relative_S.embedding = tf.complex(self.relative_S.embedding_r, self.relative_S.embedding_i)
        
        query = tf.expand_dims(query, axis=1)
        value = tf.expand_dims(value, axis=1)
        key   = tf.expand_dims(key, axis=1)
        heads = self._get_relative_attention(query, value, key)
        W = tf.complex(self.W_r, self.W_i)
        outputs = tf.reduce_sum(tf.matmul(heads, W), axis=1)
        
        return outputs

#%%
class LayerScale(tf.keras.layers.Layer):
    # per-channel reweighting
    def __init__(self, epsilon=1e-4, gamma_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma_initializer = tf.keras.initializers.Constant(epsilon)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
    
    def build(self, input_shape):
        shape = input_shape[-1:]
        self.gamma = self.add_weight(
            name="gamma",
            shape=shape,
            initializer=self.gamma_initializer, 
            regularizer=self.gamma_regularizer,
            trainable=True
            )
    
    @tf.function
    def call(self, inputs):
        outputs = tf.matmul(inputs, tf.expand_dims(tf.linalg.diag(self.gamma), 0))
        return outputs

class ComplexLayerScale(tf.keras.layers.Layer):
    # per-channel reweighting with complex numbers
    def __init__(self, epsilon=1e-4, gamma_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma_initializer = tf.keras.initializers.Constant(epsilon)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        
    
    def build(self, input_shape):
        shape = input_shape[-1:]
        self.gamma_r = self.add_weight(
            name="gamma real",
            shape=shape,
            initializer=copy_class(self.gamma_initializer),
            regularizer=self.gamma_regularizer,
            trainable=True
            )
        self.gamma_i = self.add_weight(
            name="gamma imag",
            shape=shape,
            initializer=copy_class(self.gamma_initializer), 
            regularizer=self.gamma_regularizer,
            trainable=True
            )
    
    @tf.function
    def call(self, inputs, training=None):
        diag_gamma = tf.linalg.diag(tf.complex(self.gamma_r, self.gamma_i))
        outputs = tf.matmul(inputs, tf.expand_dims(diag_gamma, 0))
        return outputs

#%%
class BaseNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 epsilon=0.001,
                 center=True,
                 scale=True,
                 gamma_initializer="ones",
                 beta_initializer="zeros",
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 **kwargs):

        super().__init__(**kwargs)
        if isinstance(axis, int):
            axis = [axis]
        self.axis = list(axis)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)

    def build(self, input_shape):
        self.axis = [len(input_shape) + ax if ax < 0 else ax for ax in self.axis]
        self.shape = tuple(input_shape[ax] for ax in self.axis)
        self.para_shape = tuple(input_shape[ax] if ax in self.axis else 1 for ax in range(1, len(input_shape)))
    
    def build_weight(self, add_name=""):
        gamma, beta = 1., 0.
        if self.scale:
            gamma = self.add_weight(
                name=f'gamma{add_name}',
                shape=self.shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                trainable=True
            )
        if self.center:
            beta = self.add_weight(
                name=f'beta{add_name}',
                shape=self.shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                trainable=True
            )
        
        return gamma, beta

    @tf.function
    def _get_normalize(self, inputs, training=None):        
        mean = tf.math.reduce_mean(inputs, axis=self.reduce_axis, keepdims=True)
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), axis=self.reduce_axis, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= tf.reshape(self.gamma, self.para_shape)
        if self.center:
            outputs += tf.reshape(self.beta, self.para_shape)
        return outputs

#%%
class LayerNormalization(BaseNormalization):
    '''
        axis: at which reduce dimension; different from InstanceNormalization
        1. CNN : compute statistics across (Height, Width, Channel)
        e.g. input_shape = (Batch, Height, Width, Channel); axis=(1,2,3)
        2. RNN : compute statistics across (Channel) ONLY but NOT (Time)
        e.g. input_shape = (Batch, Time, Frequency, Channel); axis=(2,3)
        3. Use TimeDistributed for (Batch, Time, Height, Width, Channel) and see CNN
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.gamma, self.beta = self.build_weight()
        self.reduce_axis = self.axis                # Only difference against InstanceNormalization
    
    @tf.function
    def call(self, inputs, training=None):
        outputs = self._get_normalize(inputs, training)
        
        return outputs

class ComplexLayerNormalization(BaseNormalization):
    '''
        axis: at which reduce dimension; different from InstanceNormalization
        1. CNN : compute statistics across (Height, Width, Channel)
        e.g. input_shape = (Batch, Height, Width, Channel); axis=(1,2,3)
        2. RNN : compute statistics across (Channel) ONLY but NOT (Time)
        e.g. input_shape = (Batch, Time, Frequency, Channel); axis=(2,3)
        3. Use TimeDistributed for (Batch, Time, Height, Width, Channel) and see CNN
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.gamma_r, self.beta_r = self.build_weight(add_name=" real")
        self.gamma_i, self.beta_i = self.build_weight(add_name=" imag")
        self.reduce_axis = self.axis                # Only difference against InstanceNormalization
    
    @tf.function
    def call(self, inputs, training=None):
        self.gamma = tf.complex(self.gamma_r, self.gamma_i)
        self.beta = tf.complex(self.beta_r, self.beta_i)
        outputs = self._get_normalize(inputs, training)
        
        return outputs


#%%
class InstanceNormalization(BaseNormalization):
    '''
        axis: at which channel locate; different from LayerNormalization
        compute statistics across (Height, Width) ONLY
        e.g. input_shape = (Batch, Height, Width, Channel); axis=-1
        e.g. input_shape = (Batch, Time, Frequency, Channel); axis=-1
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

    def build(self, input_shape):
        super().build(input_shape)
        self.gamma, self.beta = self.build_weight()
        self.reduce_axis = [ax for ax in range(1, len(input_shape)) if ax not in self.axis]

    @tf.function
    def call(self, inputs, training=None):
        outputs = self._get_normalize(inputs, training)
        
        return outputs
    
class ComplexInstanceNormalization(BaseNormalization):
    '''
        axis: at which channel locate; different from LayerNormalization
        compute statistics across (Height, Width) ONLY
        e.g. input_shape = (Batch, Height, Width, Channel); axis=-1
        e.g. input_shape = (Batch, Time, Frequency, Channel); axis=-1
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

    def build(self, input_shape):
        super().build(input_shape)
        self.gamma_r, self.beta_r = self.build_weight(add_name=" real")
        self.gamma_i, self.beta_i = self.build_weight(add_name=" imag")
        self.reduce_axis = [ax for ax in range(1, len(input_shape)) if ax not in self.axis]

    @tf.function
    def call(self, inputs, training=None):
        self.gamma = tf.complex(self.gamma_r, self.gamma_i)
        self.beta = tf.complex(self.beta_r, self.beta_i)
        outputs = self._get_normalize(inputs, training)
        
        return outputs