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

from tqdm import tqdm
from skimage.io import imsave
import numpy as np
from config import *
import os

def save_png(images, col_size, path):

    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * col_size[0], w * col_size[1], 3))
    for idx, image in enumerate(images):
        i = idx % col_size[1]
        j = idx // col_size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    root, file_name = os.path.split(path)
    if not os.path.exists(root):
        os.makedirs(root)
    #print(np.max(merge_img), np.min(merge_img))
    merge_img = (merge_img * 255 / 2 + 255 / 2).clip(0, 255).astype(np.uint8)
    imsave(path, merge_img)
    #print(np.max(merge_img), np.min(merge_img))

def save_tiff(images, col_size, path):

    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * col_size[0], w * col_size[1], 1))
    for idx, image in enumerate(images):
        i = idx % col_size[1]
        j = idx // col_size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    root, file_name = os.path.split(path)
    if not os.path.exists(root):
        os.makedirs(root)

    imsave(path, np.reshape(merge_img[:,:,0], [merge_img.shape[0],merge_img.shape[1]]).astype(np.float32))

def save_one_tiff(images, path):

    img = (images + 1.0) / 2.0

    root, file_name = os.path.split(path)
    if not os.path.exists(root):
        os.makedirs(root)

    imsave(path, img.astype(np.float32))

def save_one_png(images, path):
    root, file_name = os.path.split(path)
    if not os.path.exists(root):
        os.makedirs(root)

    imsave(path, images)

def read_label_text_files(text_file_path):
    label_dict = {}
    for cls in class_table:
        with open(text_file_path.format(cls), 'r') as f:
            for line in f:
                line = line.replace('\n','').replace('  ',' ')
                file_name, is_this_cls = line.split(' ')
                file_path = os.path.join(data_folder, file_name+'.jpg')

                if is_this_cls == '0' or is_this_cls == '1':
                    if file_path in label_dict:
                        label_dict[file_path][class_table[cls]] = 1
                    else:
                        label_dict[file_path] = [0]*num_class
                        label_dict[file_path][class_table[cls]] = 1
    return label_dict

def convert_label_dict_to_lists(dict):
    img_path_list = []
    label_list = []
    for img_path in dict:
        img_path_list.append(img_path)
        label_list.append(dict[img_path])

    return img_path_list, label_list

def get_class_num_by_index(idx):
    for class_name in class_table:
        if class_table[class_name] == idx:
            return class_name
