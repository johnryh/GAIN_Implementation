'''
Copyright 2018 - 2019 Duks University
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License Version 2 as published by
the Free Software Foundation.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License Version 2
along with this program.  If not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt>.
'''


from config import *
from network_utility import *
import tensorflow as tf
from resnet_v1 import *

def average_gradients(tower_grads):
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

def get_trainable_weights():
    var_list = []
    for v in tf.trainable_variables():
        if 'block4' in v.name or 'logits' in v.name:
            var_list.append(v)

    return var_list


def initialize_optimizer(scope, model_lr):
    with tf.variable_scope(scope):
        return tf.train.AdamOptimizer(learning_rate=model_lr, beta1=0.9, beta2=0.99, epsilon=1e-8, name='optim')


# print out the number of parameters in a given scope
def print_model_stats(scope):
    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('{} --- Total parameters:{}M'.format(scope, total_parameters/1e6))


def get_var_dict_by_scope(scope):
    var_dict = {}
    for v in tf.global_variables():
        if scope in v.name and '/logits/' not in v.name:
            var_dict[v.name.replace('GAIN/','').replace(':0','')] = v
            #print('fe_scope:',v.name.replace('GAIN/','').replace(':0',''))

    return var_dict


class GAIN():
    def __init__(self, img_batch, label_batch):
        self.img_batch = img_batch
        self.label_batch = label_batch
        #self.ln_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')

        self.optimizer = initialize_optimizer('Optimizer', lr_rate)
        print('Solver Configured')

        self.build_model()
        print_model_stats('GAIN')


    def build_model(self):
        img_split = tf.split(self.img_batch, num_gpus, name='img_split') if num_gpus > 1 else [self.img_batch]
        label_split = tf.split(self.label_batch, num_gpus, name='label_split') if num_gpus > 1 else [self.label_batch]
        tower_cls_loss = []; tower_grads = []
        tower_score = []; tower_accuracy = []; tower_auc = []; tower_CAM = []; tower_logits = []; tower_am_loss = []
        for gpu_id in range(num_gpus):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope('GAIN'):
                    with slim.arg_scope(resnet_arg_scope()):
                        logits, end_features = resnet_v1_50(inputs=img_split[0], num_classes=num_class, is_training=True, scope=feature_extractor_scope, reuse=tf.AUTO_REUSE)
                        tower_logits.append(logits)

                    with tf.variable_scope('Loss/Classification_Loss'):
                        tower_cls_loss.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_split[gpu_id], logits=logits)))

                    with tf.variable_scope('CAM'):
                        feature_gradients = []; importance_weights = []; attention_maps = []

                        #fe = end_features['GAIN/resnet_v1_50/block4/unit_2/bottleneck_v1']
                        fe = end_features['GAIN/resnet_v1_50/block4/unit_2/bottleneck_v1']

                        with tf.variable_scope('Stack_CAMs'):
                            for logit_idx in range(num_class):

                                # compute gradient of each logit w.r.t chosedn end_features
                                feature_gradients.append(tf.gradients(logits[:,logit_idx:logit_idx+1], fe)[0])
                                #feature_gradients[-1] = tf.Print(feature_gradients[-1], [tf.reduce_mean(feature_gradients[-1][0,:,:,0]), feature_gradients[-1][0,:,:,0]], summarize=100)

                                # apply GAP to compute importance importance weights of each features
                                importance_weights.append(tf.reduce_mean(feature_gradients[-1],  axis=[1,2], keepdims=True))

                                # compute attention maps by multiplying the importance weight of each feature map to itself
                                attention_maps.append(importance_weights[-1]*feature_gradients[-1])
                                #attention_maps.append(feature_gradients[-1])

                            attention_map_stack = tf.concat(attention_maps, axis=3)
                            CAM = tf.nn.relu(tf.reduce_sum(attention_map_stack, axis=3, keepdims=True))
                            tower_CAM.append(CAM)

                            soft_mask = sigmoid_threshold(CAM, w=1, alpha=0.1)
                            soft_mask = tf.image.resize_images(soft_mask, [train_img_h, train_img_w], method=tf.image.ResizeMethod.BILINEAR)
                            masked_image = img_split[0] - tf.multiply(img_split[0], soft_mask)

                    with slim.arg_scope(resnet_arg_scope()):
                        attention_mining_logits, _ = resnet_v1_50(inputs=masked_image, num_classes=num_class, is_training=True, scope=feature_extractor_scope, reuse=tf.AUTO_REUSE)

                    with tf.variable_scope('Loss/Attention_Mining_Loss'):
                        tower_am_loss.append(am_loss_weight * tf.reduce_sum(tf.nn.sigmoid(attention_mining_logits)/num_class))
                    with tf.variable_scope('Loss'):
                        total_loss = tower_cls_loss[-1] + tower_am_loss[-1]
                        tower_grads.append(self.optimizer.compute_gradients(total_loss, var_list=get_trainable_weights()))

        with tf.variable_scope('Savers'):
            self.saver = tf.train.Saver(name='Saver', max_to_keep=None)
            self.loader = tf.train.Saver(name='FE_Loader', var_list=get_var_dict_by_scope(feature_extractor_scope))

        with tf.variable_scope('Apply_Optim_Gradients'), tf.device("/device:{}:1".format(controller)):
            self.grads = average_gradients(tower_grads)
            self.apply_grads = self.optimizer.apply_gradients(self.grads, name='Apply_Grads')

        with tf.variable_scope('Reporting'):
            self.cls_loss = tf.reduce_mean(tower_cls_loss)
            self.am_loss = tf.reduce_mean(tower_am_loss)
            self.total_loss = self.cls_loss + self.am_loss
            self.logits = tf.nn.sigmoid(tf.concat(tower_logits, axis=0))
            self.CAM = tf.concat(tower_CAM, axis=0)

            tf.summary.scalar('cls_loss loss', self.cls_loss)
            tf.summary.scalar('am_loss loss', self.am_loss)
            tf.summary.scalar('total loss', self.total_loss)
