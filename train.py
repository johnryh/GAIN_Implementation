from Input_Pipeline import *
from build_models import *
from utilities import *

from tqdm import tqdm
import numpy as np
import os, time
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

prev_phase_iter = 0
if __name__ == '__main__':

    tf.reset_default_graph()

    train_label_dict = read_label_text_files(r'C:\projects_yinhao\data\VOC2012\ImageSets\Main\{}_train.txt')
    val_label_dict = read_label_text_files(r'C:\projects_yinhao\data\VOC2012\ImageSets\Main\{}_val.txt')

    print('training images: {}'.format(len(train_label_dict)))
    print('validation images: {}'.format(len(val_label_dict)))

    train_img_path_list, train_label_list = convert_label_dict_to_lists(train_label_dict)
    val_img_path_list, val_label_list = convert_label_dict_to_lists(val_label_dict)

    with tf.variable_scope('training_input'):
        img_batch, label_batch = build_input_pipline(batch_size, train_img_path_list, train_label_list)
        img_batch = tf.reshape(img_batch, [per_gpu_size, train_img_h, train_img_w, 3])
        label_batch = tf.reshape(label_batch, [per_gpu_size, num_class])

    with tf.variable_scope('testing_input'):
        val_img_batch, val_label_batch = build_input_pipline(batch_size, val_img_path_list, val_label_list)


    model = GAIN(img_batch, label_batch)


    merged = tf.summary.merge_all()

    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        train_writer = tf.summary.FileWriter('runs/{}/logs/train/'.format(exp_name), sess.graph)
        val_writer = tf.summary.FileWriter('runs/{}/logs/val/'.format(exp_name), sess.graph)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        print('Session Initiated')

        model.loader.restore(sess, feature_extractor_ckpt_path)
        print('FE Weights Loaded: {}'.format(feature_extractor_ckpt_path))


        per_iter_time = 0
        with tqdm(total=int(epoch_num * total_samples / batch_size-prev_phase_iter), unit='it') as pbar:
            train_start_time = time.time()
            plt.ion()
            for iter in range(prev_phase_iter,int(epoch_num * total_samples / batch_size)):
                iter_start_time = time.time()

                _, summary = sess.run([model.apply_grads, merged])
                iter_per_sec = 1/(time.time() - iter_start_time)
                train_writer.add_summary(summary, iter)

                if iter % 10 == 0: pbar.set_postfix({'it_ins/s':'{:4.2f}, CLS_Loss:{}, AM_Loss:{}'.format(iter_per_sec, 0, 0)})
                pbar.update(1)


                if iter % 100 == 0:
                    val_img_batch_l, val_label_batch_l = sess.run([val_img_batch, val_label_batch])

                    val_img_l, val_label_l, cam_l, predictions, val_summary = sess.run([model.img_batch, model.label_batch, model.CAM, model.logits, merged], feed_dict={model.img_batch:val_img_batch_l, model.label_batch:val_label_batch_l})
                    val_writer.add_summary(val_summary, iter)
                    top5_cls_indxes = (np.argpartition(predictions[0,:], -3)[-3:])
                    print(top5_cls_indxes)
                    pred_class_names = [get_class_num_by_index(idx) for idx in top5_cls_indxes]
                    preds_str = ''

                    for i in range(3):
                        preds_str += '{}:{:.1f}    '.format(pred_class_names[i], 100*predictions[0][top5_cls_indxes[i]])

                    plt.figure(1)
                    plt.imshow(cam_l[0,:,:,0])
                    plt.figure(2)
                    plt.imshow(val_img_l[0,:,:,:])
                    plt.title(preds_str)
                    plt.pause(5)