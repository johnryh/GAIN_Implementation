
from config import *
import tensorflow as tf
import numpy as np
import os, math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


def PN(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)

    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))


def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[3], s[1], s[2]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.

        return tf.concat([x, y], axis=3)                        # [NCHW]  Append as new fmap.


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


def conv2d_transpose(z, kernel_num, kernel_size, stride, padding='SAME', isTrain=True, activation='leaky_relu', name='kernel', batch_norm=False, factor=2, use_PN=False, gain=np.sqrt(2)):
    with tf.variable_scope(name):
        with tf.device("/device:{}:0".format(controller)):
            initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32, factor=np.sqrt(2))
            #kernel = tf.get_variable('weights', [kernel_size, kernel_size, int(kernel_num), z.shape[-1].value], initializer=initializer)
            weights = get_weight([kernel_size, kernel_size, int(kernel_num), z.shape[-1].value], gain, use_wscale=True, fan_in=(kernel_size**2)*z.shape[1].value)
            biases = tf.get_variable('bias', [int(kernel_num)], initializer=initializer)

            shape = tf.constant([z.shape[0].value, z.shape[1].value*factor, z.shape[2].value*factor, kernel_num])

        conv = tf.add(tf.nn.conv2d_transpose(z, weights, shape, strides=[1, stride[0], stride[1], 1], padding=padding), biases)

        if batch_norm:
            conv = tf.layers.batch_normalization(conv, training=isTrain)

        out = activation_func(conv, activation)

        if use_PN:
            out = PN(out)

        return out


def input_projection(x, num_base_features, batch_norm=False):
    with tf.variable_scope('Input_Projection'):
        if batch_norm:
            x = tf.layers.batch_normalization(x, training=True)

        latent = dense(x, units=4 * 4 * num_base_features, activation='leaky_relu', name='Dense_1', gain=np.sqrt(2)/4, use_PN=False)
        latent = tf.reshape(latent, [-1, 4, 4, num_base_features])

        latent = conv2d(latent, 0, num_kernal=g_max_num_features, kernal_size=3, stride=1, padding='SAME', scope='Projection_Conv', use_PN=True)

        return latent


def rgb_projection(fetures, input_dim, name='output_projection', is_smooth=False):
    with tf.variable_scope(name):
        #fetures = conv2d(fetures, input_dim, num_kernal=max(int(int(fetures.shape[-1])/4), 16), kernal_size=3, stride=1, padding='SAME', scope='reduced_features', activation='leaky_relu', use_PN=True)
        output = conv2d(fetures, input_dim, num_kernal=3, kernal_size=1, stride=1, padding='SAME', scope='to_rgb', activation=to_rgb_activation, gain=1)

        #output = tf.clip_by_value(output, 0, 1)
    return output


def from_rgb_mask(image, mask, num_features, name):
    with tf.variable_scope(name):
        mask_features = conv2d(mask, 1, 3, 'mask_conv', kernal_size=3, stride=1, padding='SAME')
        x = tf.concat([image, mask_features], axis=3)
        features = conv2d(x, -1, num_features, 'to_feature_space', kernal_size=3, stride=1, padding='SAME')


        return features


def score_projection(features_volume, num_featrues_list, name='score_projection'):
    with tf.variable_scope(name):
        reg1 = conv2d(features_volume, -1, num_featrues_list[0], 'conv_reg1_{}'.format(1), kernal_size=(3, 3),stride=1, padding='SAME')

        with tf.variable_scope('Flatten'):
            flatten_features = tf.reshape(reg1, [int(batch_size/num_gpus), -1])

        latent_feature = dense(flatten_features, units=num_featrues_list[1], name='Dense_0', activation='linear', gain=1)
        score = dense(latent_feature, units=1, name='Dense', activation='linear', gain=1)

        return score


def mask_projection_path_more_features(fake_masks):
    #num_mask_features = {0:8, 1:8, 2:16, 3:16, 4:32, 5:32, 6:32, 7:32, 8:32}
    num_mask_features = {0:32, 1:32, 2:32, 3:16, 4:16, 5:8, 6:8, 7:8, 8:8}

    use_PN = False
    mask_features = [fake_masks]
    out_features = [fake_masks]

    target_size = 16
    if len(mask_features) == 1:
        curr_block_phase = int(math.log(int(mask_features[-1].shape[1]) / 4, 2))
        with tf.variable_scope('mask_to_fetures'):
            mask_features.append(conv2d(mask_features[-1], mask_features[-1].shape[3], num_kernal=8, scope='mask_fetures', kernal_size=8, stride=1, use_PN=use_PN, padding='SAME'))
            mask_features.append(conv2d(mask_features[-1], mask_features[-1].shape[3], num_kernal=num_mask_features[curr_block_phase], scope='mask_fetures_l', kernal_size=8, stride=1, use_PN=use_PN, padding='SAME'))

            num_concat_features = 8 if curr_block_phase >= 4 else 8

            out_features.append(mask_features[-1][:, :, :, :num_concat_features])


    while mask_features[-1].shape[1] > target_size:
        curr_block_phase = int(math.log(int(mask_features[-1].shape[1]) / 4, 2))-1
        with tf.variable_scope('mask_{}'.format(curr_block_phase-1)):
            mask_features.append(conv2d(mask_features[-1], mask_features[-1].shape[3], num_kernal=num_mask_features[curr_block_phase], scope='mask_conv_{}'.format(curr_block_phase-1), kernal_size=4, stride=1, use_PN=use_PN, padding='SAME'))
            mask_features.append(conv2d(mask_features[-1], mask_features[-1].shape[3], num_kernal=num_mask_features[curr_block_phase], scope='mask_conv_{}_l'.format(curr_block_phase-1), kernal_size=4, stride=2, use_PN=use_PN, padding='SAME'))

            num_concat_features = 8 if  curr_block_phase >= 4 else 8

            out_features.append(mask_features[-1][:, :, :, :num_concat_features])

    if use_embedding:
        with tf.variable_scope('Mask_Embedding'):
            mask_features.append(dense(tf.reshape(mask_features[-1], [int(batch_size/num_gpus), target_size * target_size * mask_features[-1].shape[-1]]), embedding_latent_0_size, activation='linear', name='embedding_latent_0_size', gain=1))
            #mask_features.append(dense(mask_features[-1], embedding_latent_1_size, activation='linear', name='embedding_latent_1_size', gain=1))
            mask_features.append(dense(mask_features[-1], embedding_size, name='embedding_out', gain=np.sqrt(2)/4))
            out_features.append(mask_features[-1])


    return out_features, mask_features


def get_z(batch_size, z_length):
    with tf.variable_scope('z'):
        z = tf.random_normal(shape=[batch_size, z_length], mean=0, stddev=1, name='random_z')

    return z


class GAIN():

    def __init__(self, latent, real_img, mask, phase, LAMBDA):
        self.initialize_optimizer()
        self.build_model(latent, real_img,  mask, phase=phase, LAMBDA=LAMBDA)
        self.model_stats()

    def initialize_optimizer(self):
        with tf.variable_scope('GAN_Optimizer'):
            with tf.variable_scope('Optim'):
                self.optim = tf.train.AdamOptimizer(learning_rate=d_ln_rate, beta1=0.0, beta2=0.99, epsilon=1e-8, name='Optim') # was 0.5, 0.9
        print('Solver Configured')

    def build_model(self, latent, real_img, mask, phase, LAMBDA):
        latent_split = tf.split(latent, num_gpus, name='latent_split') if num_gpus > 1 else [latent]
        real_img_split = tf.split(real_img, num_gpus, name='real_img_split') if num_gpus > 1 else [real_img]
        mask_split = tf.split(mask, num_gpus, name='mask_split') if num_gpus > 1 else [mask]


        tower_logits=[]
        tower_loss=[]
        tower_grads = []
        self.real_images = real_img
        self.fake_masks = mask
        for gpu_id in range(num_gpus):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope('GAIN') as GAN_scope:
                    if gpu_id == 0:
                        # trainable variables for each network
                        T_vars = tf.trainable_variables()
                        print('var_list initiated')

                        tf.summary.scalar('g_loss/fake_score', g_loss)
                        tf.summary.scalar('d_loss', d_loss)
                        tf.summary.scalar('real_score', tf.reduce_mean(real_score))
                        tf.summary.scalar('alpha', self.g_alpha)
                        tf.summary.scalar('gradient_penalty', gradient_penalty)
                        tf.summary.scalar('d_drift_penalty', d_drift_penalty)

                        self.saver = tf.train.Saver(name='P{}_Saver'.format(phase), max_to_keep=None)

                    with tf.variable_scope('Compute_Optim_Gradients'):
                        tower_grads.append(self.d_optim.compute_gradients(loss, var_list=T_vars))

                tower_loss.append(loss)
                tower_logits.append(logits)

        with tf.variable_scope('Sync_Point'):
            self.loss = tf.reduce_mean(tower_loss, axis=0, name='g_loss')
            self.logits = tf.concat(tower_logits, axis=0)

        with tf.variable_scope('Solver'):

            with tf.variable_scope('Apply_Optim_Gradients'), tf.device("/device:{}:1".format(controller)):
                self.grads = self.average_gradients(tower_grads)

                self.apply_g_grad = self.Optim.apply_gradients(self.grads, name='Apply_Grads')

    def average_gradients(self, tower_grads):
        with tf.variable_scope('Average_Gradients'):

            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = [g for g, _ in grad_and_vars]
                grad = tf.reduce_mean(grads, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)

                average_grads.append(grad_and_var)
            return average_grads

    def model_stats(self):
        total_parameters = 0
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GAIN'):
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('GAIN Total parameters:{}M'.format(total_parameters / 1e6))