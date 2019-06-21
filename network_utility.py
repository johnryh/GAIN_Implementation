from config import *
import tensorflow as tf
import numpy as np
import os, math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def sigmoid_threshold(feature, alpha, w):
    with tf.variable_scope('sigmoid_threshold'):
        return tf.nn.sigmoid(-w *(feature - alpha))

def activation_func(z, activation='leaky_relu'):
    if activation == 'leaky_relu':
        return tf.nn.leaky_relu(z)
    elif activation == 'relu':
        return tf.nn.relu(z)
    elif activation == 'selu':
        return tf.nn.selu(z)
    elif activation == 'tanh':
        return tf.nn.tanh(z)
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(z)
    elif activation == 'linear':
        return z

    assert False, 'Activation Func "{}" not Found'.format(activation)


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)

    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))


def dense(z, units, activation=None, name='Dense', gain=np.sqrt(2)/4, use_PN=False):
    with tf.variable_scope(name):
        with tf.device("/device:{}:0".format(controller)):
            assert len(z.shape) == 2, 'Input Dimension must be rank 2, but is rank {}'.format(len(z.shape))
            initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32, factor=gain)
            weights = get_weight([z.shape[1].value, units], gain, use_wscale=True)
            biases = tf.get_variable('bias', [units], initializer=initializer)

            y = tf.add(tf.matmul(z, weights), biases)

            if activation:
                y= activation_func(y, activation)

            if use_PN:
                y = PN(y)

            return y


def conv2d(input_vol, input_dim, num_kernal, scope, kernal_size=3, stride=1, activation='leaky_relu', padding='SAME', batch_norm=False, gain=np.sqrt(2), use_PN=False):
    with tf.variable_scope(scope):
        if isinstance(kernal_size, int):
            kernal_height = kernal_size
            kernal_width = kernal_size
        else:
            kernal_height = kernal_size[0]
            kernal_width = kernal_size[1]

        with tf.device("/device:{}:0".format(controller)):
            initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32, factor=gain)
            weights = get_weight([kernal_height, kernal_width, int(input_vol.shape[-1]), int(num_kernal)], gain, use_wscale=True)
            biases = tf.get_variable('bias', [int(num_kernal)], initializer=initializer)

        conv = tf.add(tf.nn.conv2d(input_vol, weights, [1, stride, stride, 1], padding=padding), biases)

        if batch_norm:
            conv = tf.layers.batch_normalization(conv, training=True)

        out = activation_func(conv, activation)

        if use_PN:
            out = PN(out)

        return out


def get_z(batch_size, z_length):
    with tf.variable_scope('z'):
        z = tf.random_normal(shape=[batch_size, z_length], mean=0, stddev=1, name='random_z')

    return z