import tensorflow as tf

def avail_activations(_return_=False):

    '''
    Prints available activations for training.

    Args:
        _return_(bool): Whether to returns the dictionary. Default: False 
    '''

    avail_acts={
        'relu': tf.nn.relu,
        'tanh': tf.nn.tanh,
        'sigmoid': tf.nn.sigmoid,
        'lrelu': tf.nn.leaky_relu,
        'elu': tf.nn.elu,
        'exponential': tf.keras.activations.exponential,
        'hard_sigmoid': tf.keras.activations.hard_sigmoid,
        'linear': tf.keras.activations.linear,
        'selu': tf.keras.activations.selu,
        'softmax': tf.nn.softmax,
        'softplus': tf.nn.softplus,
        'softsign': tf.nn.softsign
    }

    if not _return_:
        for acti in list(avail_acts):
            print(acti)

    else:
        return avail_acts


def avail_initializers(_return_=False):

    '''
    Prints available initializers for training.

    Args:
        _return_(bool): Whether to returns the dictionary. Default: False 
    '''

    avail_init={
        'zeros': tf.keras.initializers.Zeros(),
        'ones': tf.keras.initializers.Ones(),
        'constant': tf.keras.initializers.Constant(),
        'random_normal': tf.keras.initializers.RandomNormal(),
        'random_uniform': tf.keras.initializers.RandomUniform(),
        'truncated_normal': tf.keras.initializers.TruncatedNormal(),
        'variance_scaling': tf.keras.initializers.VarianceScaling(),
        'orthogonal': tf.keras.initializers.Orthogonal(),
        'identity': tf.keras.initializers.Identity(),
        'lecun_uniform': tf.keras.initializers.lecun_uniform(),
        'lecum_normal': tf.keras.initializers.lecun_normal(),
        'glorot_normal': tf.keras.initializers.glorot_normal(),
        'glorot_uniform': tf.keras.initializers.glorot_uniform(),
        'he_normal': tf.keras.initializers.he_normal(),
        'he_uniform': tf.keras.initializers.he_uniform(),
        'xavier': tf.contrib.layers.xavier_initializer()
    }
    
    
    if not _return_:
        for init in list(avail_init):
            print(init)

    else:
        return avail_init


def avail_optimizers(_return_=False):

    '''
    Prints available optimizers for training.

    Args:
        _return_(bool): Whether to returns the dictionary. Default: False 
    '''

    avail_opts={
        'ftrl': tf.train.FtrlOptimizer,
        'adam': tf.compat.v1.train.AdamOptimizer,
        'adagrad': tf.compat.v1.train.AdagradOptimizer,
        'rmsprop': tf.compat.v1.train.RMSPropOptimizer,
        'adadelta': tf.compat.v1.train.AdadeltaOptimizer,
        'momentum': tf.compat.v1.train.MomentumOptimizer,
        'adagradDA': tf.compat.v1.train.AdagradDAOptimizer,
        'gradient_descent': tf.compat.v1.train.GradientDescentOptimizer,
        'proximal_adagrad': tf.compat.v1.train.ProximalAdagradOptimizer,
        'proximal_gradient_descent': tf.compat.v1.train.ProximalGradientDescentOptimizer
    }
    
    
    if not _return_:
        for opt in list(avail_opts):
            print(opt)

    else:
        return avail_opts


def avail_regularizers(_return_=False):

    '''
    Prints available regularizers for training.

    Args:
        _return_(bool): Whether to returns the dictionary. Default: False 
    '''

    avail_regul={
        'l1': tf.keras.regularizers.l1(),
        'l2': tf.keras.regularizers.l2(),
        'l1_l2': tf.keras.regularizers.l1_l2(),
        "None": None
    }
    
    
    if not _return_:
        for reg in list(avail_regul):
            print(reg)

    else:
        return avail_regul


def avail_metrics(_return_=False):

    '''
    Prints available loss functions for training.

    Args:
        _return_(bool): Whether to returns the dictionary. Default: False 
    '''

    avail_metrics = {
        'mse': tf.keras.losses.mean_squared_error,
        'mae': tf.keras.losses.mean_absolute_error,
        'mape': tf.keras.losses.mean_absolute_percentage_error,
        'msle': tf.keras.losses.mean_squared_logarithmic_error,
        'squ_h': tf.keras.losses.squared_hinge,
        'hinge': tf.keras.losses.hinge,
        'logcosh': tf.keras.losses.logcosh,
        'kld': tf.keras.losses.kullback_leibler_divergence,
        'poisson': tf.keras.losses.poisson,
        'co-prox': tf.keras.losses.cosine_proximity,
        'cosine': tf.keras.losses.cosine
    }

    if not _return_:
        for loss_fn in list(avail_metrics):
            print(loss_fn)

    else:
        return avail_metrics