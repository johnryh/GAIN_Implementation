from config import *

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
from utilities import *

def tf_get_crop_value(img):

    img_w = tf.cast(tf.shape(img)[1],dtype=tf.float32)
    img_h = tf.cast(tf.shape(img)[0],dtype=tf.float32)

    x1_c = tf.reshape(tf.cast(tf.random_uniform([1], minval=0, maxval=img_w*0.1),dtype=tf.int32),[])
    x2_c = tf.reshape(tf.cast(tf.random_uniform([1], minval=0, maxval=img_w*0.1),dtype=tf.int32),[])
    y1_c = tf.reshape(tf.cast(tf.random_uniform([1], minval=0, maxval=img_h*0.1),dtype=tf.int32),[])
    y2_c = tf.reshape(tf.cast(tf.random_uniform([1], minval=0, maxval=img_h*0.1),dtype=tf.int32),[])

    return [x1_c, x2_c, y1_c, y2_c]


def tf_crop(img):
    img_w = tf.cast(tf.shape(img)[1],dtype=tf.int32)
    img_h = tf.cast(tf.shape(img)[0],dtype=tf.int32)
    x1_c, x2_c, y1_c, y2_c = tf_get_crop_value(img)

    img = img[y1_c:img_h-y2_c, x1_c:img_w-x2_c]

    return img


def parse_func(img_path, label):

    img_file = tf.read_file(img_path)

    img = tf.image.decode_jpeg(img_file, channels=3)

    #cast to proper type
    img = tf.cast(img, dtype=tf.float32)

    img = tf_crop(img)

    #downsample to proper size for depeneds on phase
    img = tf.image.resize_images(img, [train_img_h, train_img_w])
    img = img/255

    img = flip(img)

    label = tf.cast(label, tf.float32)

    return img, label


def flip(img):
    do_flip = tf.reshape(tf.random_uniform([1], minval=0, maxval=1),[]) > tf.reshape(FLIP_RATE,[])
    img = tf.cond(do_flip, lambda:tf.image.flip_left_right(img), lambda: img)

    return img


def build_input_pipline(batch_size, img_path_list, label_list):
    with tf.variable_scope('Input_Pipeline'):

        ds_train = tf.data.Dataset.from_tensor_slices((img_path_list, label_list)).shuffle(6000)
        ds_train = ds_train.map(parse_func, num_parallel_calls=12)

        ds_train = ds_train.repeat().batch(batch_size).prefetch(batch_size * 3) # add shuffling
        iterator_train = ds_train.make_one_shot_iterator()

        return iterator_train.get_next()




train_label_dict = read_label_text_files(r'C:\projects_yinhao\data\VOC2012\ImageSets\Main\{}_train.txt')
val_label_dict = read_label_text_files(r'C:\projects_yinhao\data\VOC2012\ImageSets\Main\{}_val.txt')

print('training images: {}'.format(len(train_label_dict)))
print('validation images: {}'.format(len(val_label_dict)))

train_img_path_list, train_label_list = convert_label_dict_to_lists(train_label_dict)
val_img_path_list, val_label_list = convert_label_dict_to_lists(val_label_dict)
if __name__ == '__main__':
    img_batch, label_batch = build_input_pipline(batch_size, train_img_path_list, train_label_list)
    config = tf.ConfigProto(allow_soft_placement=True)
    plt.ion()

    with tf.Session(config=config) as sess:
        for _ in tqdm(range(10000)):
            img_l, label_l = sess.run([img_batch, label_batch])
            if label_l[0,11] == 1:
                img_l = img_l[0,:,:,:].astype(np.uint8)
                print(label_l[0,:])
                plt.figure(1)
                plt.imshow(img_l)
                plt.pause(0.5)