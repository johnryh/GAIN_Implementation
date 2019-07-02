
class_table = {'aeroplane': 0,
               'bicycle': 1,
               'bird': 2,
               'boat': 3,
               'bottle': 4,
               'bus': 5,
               'car': 6,
               'cat': 7,
               'chair': 8,
               'cow': 9,
               'diningtable': 10,
               'dog': 11,
               'horse': 12,
               'motorbike': 13,
               'person': 14,
               'pottedplant': 15,
               'sheep': 16,
               'sofa': 17,
               'train': 18,
               'tvmonitor': 19}

controller = 'CPU'

num_gpus = 1
epoch_num = 35
batch_size = 4

num_class = 20
train_img_h = 512
train_img_w = 512

lr_rate = 0.00001
am_loss_weight = 1

FLIP_RATE = 0.5

per_gpu_size = int(batch_size/num_gpus)

data_folder = r'..\..\data\VOC2012\JPEGImages'

exp_name = 'GAIN_weak_supervision'
feature_extractor_scope = 'resnet_v1_50'
feature_extractor_ckpt_path = r'resnet_50_v1_model\resnet_v1_50.ckpt'
print('loading:',[train_img_h, train_img_w])
print('num_class: {}'.format(num_class))

total_samples = 5717# VOC total sample size
