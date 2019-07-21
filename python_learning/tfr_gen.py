# **********************************************************
# Author: Hanyu Wang(王涵玉)
# **********************************************************
# This script generates TFRecords from geometry images.

import tensorflow as tf
import os
import numpy as np
import argparse
from os.path import join
from tqdm import tqdm
from random import shuffle


parser = argparse.ArgumentParser()

parser.add_argument('--tfr_dir', '-d',
                     default='/data/yqwang/Dataset/faust_256p_045_cb/gi_TFRecords',
                    type=str, help='directory to store TFRecords')
parser.add_argument('--gi_dir', '-s',
                     default='/data/yqwang/Dataset/faust_256p_045_cb/gi_classified',
                    type=str, help='directory to read geometry images')


def open_gi(path) -> list:
    # Open a gi file and return the content as numpy.ndarray

    gis = [[[]]]

    with open(path, 'r') as text:

        blank_line_count = 0
        tensor = gis[-1]

        for line in text:
            if line == '\n':
                blank_line_count += 1
            else:
                if blank_line_count == 1:
                    tensor.append([])

                elif blank_line_count == 3:
                    gis.append([[]])
                    tensor = gis[-1]

                blank_line_count = 0
                tensor[-1].append([float(i) for i in line.strip(' \t\n').split()])

    return list(np.asarray(gis, dtype=np.float32).transpose((0, 2, 3, 1)))


if __name__ == '__main__':

    args = parser.parse_args()

    tfrecords_dir = args.tfr_dir
    gi_dir = args.gi_dir

    train_gi_dir = join(gi_dir, 'train')
    val_gi_dir = join(gi_dir, 'val')
    test_gi_dir = join(gi_dir, 'test')

    if not (os.path.exists(train_gi_dir) and os.path.exists(val_gi_dir) and os.path.exists(test_gi_dir)):
        print('ERROR: Classified geometry image path not found.')
        exit(-1)

    train_tfrecords_dir = join(tfrecords_dir, 'train')
    val_tfrecords_dir = join(tfrecords_dir, 'val')
    test_tfrecords_dir = join(tfrecords_dir, 'test')

    os.makedirs(train_tfrecords_dir, exist_ok=True)
    os.makedirs(val_tfrecords_dir, exist_ok=True)
    os.makedirs(test_tfrecords_dir, exist_ok=True)

    for type_name in tqdm(os.listdir(train_gi_dir)):
        label = int(type_name.split('_')[-1])
        # print(label)

        type_dir = join(train_gi_dir, type_name)

        train_tfrecord_path = join(train_tfrecords_dir, type_name + '.tfrecords')
        train_writer = tf.python_io.TFRecordWriter(train_tfrecord_path)

        gi_names = os.listdir(type_dir)

        gi_list = []
        for gi_name in gi_names:
            gi_path = join(type_dir, gi_name)
            gi_12rot = open_gi(gi_path)
            gi_list.extend(gi_12rot)

        np.random.shuffle(gi_list)

        for gi in gi_list:
            gi_raw = gi.tobytes()  # Convert geometry images to raw bytes.
            example = tf.train.Example(
                features=tf.train.Features(feature={

                    'gi_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gi_raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

                }))

            train_writer.write(example.SerializeToString())  # Serialize gi bytes

        train_writer.close()

    # 123213
    for type_name in tqdm(os.listdir(val_gi_dir)):
        label = int(type_name.split('_')[-1])
        # print(label)

        type_dir = join(val_gi_dir, type_name)

        val_tfrecords_path = join(val_tfrecords_dir, type_name + '.tfrecords')

        val_writer = tf.python_io.TFRecordWriter(val_tfrecords_path)

        gi_names = os.listdir(type_dir)

        gi_list = []
        for gi_name in gi_names:
            gi_path = join(type_dir, gi_name)
            gi_12rot = open_gi(gi_path)
            gi_list.extend(gi_12rot)

        np.random.shuffle(gi_list)

        for gi in gi_list:
            gi_raw = gi.tobytes()  # Convert geometry images to raw bytes.
            example = tf.train.Example(
                features=tf.train.Features(feature={

                    'gi_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gi_raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

                }))

            val_writer.write(example.SerializeToString())  # Serialize gi bytes

        val_writer.close()

    # 12312312
    for type_name in tqdm(os.listdir(test_gi_dir)):
        label = int(type_name.split('_')[-1])
        # print(label)

        type_dir = join(test_gi_dir, type_name)

        test_tfrecords_path = join(test_tfrecords_dir, type_name + '.tfrecords')

        test_writer = tf.python_io.TFRecordWriter(test_tfrecords_path)

        gi_names = os.listdir(type_dir)

        gi_list = []
        for gi_name in gi_names:
            gi_path = join(type_dir, gi_name)
            gi_12rot = open_gi(gi_path)
            gi_list.extend(gi_12rot)

        np.random.shuffle(gi_list)

        for gi in gi_list:
            gi_raw = gi.tobytes()  # Convert geometry images to raw bytes.
            example = tf.train.Example(
                features=tf.train.Features(feature={

                    'gi_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gi_raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

                }))

            test_writer.write(example.SerializeToString())  # Serialize gi bytes

        test_writer.close()



        # c = open_gi('E:/tr_scan_000_keypoint_00_rot_0.gi')
        # print(c.shape)
        # print(c.dtype)
