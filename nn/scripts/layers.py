import tensorflow as tf
import numpy
import sys, os


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_float('bn_stats_decay_factor', 0.99,
                          "moving average decay factor for stats on batch normalization")

def lrelu(x, a=0.1):
    if a < 1e-16:
        return tf.nn.relu(x,name="relu")
    else:
        return tf.maximum(x, a * x,name="relu")


def bn(x, dim, is_training=True, update_batch_stats=True, collections=None, name="bn"):
    
    with tf.name_scope(name) as scope:
        params_shape = (dim,)
        n = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
        axis = list(range(int(tf.shape(x).get_shape().as_list()[0]) - 1))
        mean = tf.reduce_mean(x, axis)
        var = tf.reduce_mean(tf.pow(x - mean, 2.0), axis)
        with tf.variable_scope("bn/"+name, reuse = tf.get_variable_scope().reuse) as scope:
            avg_mean = tf.get_variable(
                name=name + "_mean",
                shape=params_shape,
                initializer=tf.constant_initializer(0.0),
                collections=collections,
                trainable=False
            )
        
            avg_var = tf.get_variable(
                name=name + "_var",
                shape=params_shape,
                initializer=tf.constant_initializer(1.0),
                collections=collections,
                trainable=False
            )
        
            gamma = tf.get_variable(
                name=name + "_gamma",
                shape=params_shape,
                initializer=tf.constant_initializer(1.0),
                collections=collections
            )
        
            beta = tf.get_variable(
                name=name + "_beta",
                shape=params_shape,
                initializer=tf.constant_initializer(0.0),
                collections=collections,
            )
    
        if is_training:
            avg_mean_assign_op = tf.no_op()
            avg_var_assign_op = tf.no_op()
            if update_batch_stats:
                avg_mean_assign_op = tf.assign(
                    avg_mean,
                    FLAGS.bn_stats_decay_factor * avg_mean + (1 - FLAGS.bn_stats_decay_factor) * mean)
                avg_var_assign_op = tf.assign(
                    avg_var,
                    FLAGS.bn_stats_decay_factor * avg_var + (n / (n - 1))
                    * (1 - FLAGS.bn_stats_decay_factor) * var)
    
            with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
                z = (x - mean) / tf.sqrt(1e-6 + var)
        else:
            z = (x - avg_mean) / tf.sqrt(1e-6 + avg_var)
    
        return gamma * z + beta


def fc(x, dim_in, dim_out, seed=None, name='fc'):
    with tf.name_scope(name) as scope:
        num_units_in = dim_in
        num_units_out = dim_out
        weights_initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)
        
        with tf.variable_scope("fc/"+name, reuse = tf.get_variable_scope().reuse) as scope:
            weights = tf.get_variable(name + '_W',
                                    shape=[num_units_in, num_units_out],
                                    initializer=weights_initializer)
            biases = tf.get_variable(name + '_b',
                                     shape=[num_units_out],
                                     initializer=tf.constant_initializer(0.0))
        x = tf.nn.xw_plus_b(x, weights, biases)
        return x


def conv(x, ksize, stride, f_in, f_out, padding='SAME', use_bias=False, seed=None, name='conv'):
    with tf.name_scope(name) as scope:
        
        with tf.variable_scope("conv/"+name, reuse = tf.get_variable_scope().reuse) as scope:
            shape = [ksize, ksize, f_in, f_out]
            initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)
            weights = tf.get_variable(name + '_W',
                                    shape=shape,
                                    dtype='float',
                                initializer=initializer)
        x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)
    
        if use_bias:
            bias = tf.get_variable(name + '_b',
                                   shape=[f_out],
                                   dtype='float',
                                   initializer=tf.zeros_initializer)
            return tf.nn.bias_add(x, bias)
        else:
            return x

def deconv(x, ksize, stride, f_in, f_out, padding='SAME',  name='deconv',seed=None):
    
    
    
        with tf.variable_scope("deconv/"+name, reuse = tf.get_variable_scope().reuse) as scope:
            shape = [ksize, ksize, f_in, f_out]
            initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)
            weights = tf.get_variable(name + '_W',
                                    shape=shape,
                                    dtype='float',
                                initializer=initializer)

    
        out = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)
        
        return upsample(out, [2*stride,2*stride], "upsample")

def avg_pool(x, ksize=2, stride=2):
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME',name="avg_pool")


def max_pool(x, ksize=2, stride=2, name = "max_pool"):
    with tf.name_scope(name) as scope:
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME',name="max_pool")


def ce_loss(logit, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))


def ce_loss_v2(logit, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y))


def accuracy(logit, y):
    pred = tf.argmax(logit, 1)
    true = tf.argmax(y, 1)
    return tf.reduce_mean(tf.to_float(tf.equal(pred, true)))

  
def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keepdims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keepdims=True))
    return lsm
  

def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp


def entropy_y_x(logit):
    p = tf.nn.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))


''' LAYERS FOR AUTOENCODER '''

def upsample(x, factor=[2,2], name="upsample"):
    size = [int(x.shape[1] * factor[0]), int(x.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(x, size=size, align_corners=None, name=None)
    return out




