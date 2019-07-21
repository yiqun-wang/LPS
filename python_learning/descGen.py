# **********************************************************
# Author: Hanyu Wang(王涵玉)
# **********************************************************

import argparse
import os
import sys
import time
from os.path import join

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tqdm import tqdm

# In[2]:

parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpuid', '-g', default='3', type=str, metavar='N',
                    help='GPU id to run')

parser.add_argument('--batch_size', '--bs', default=512, type=int, 
                    help='batch size of evaluation')

parser.add_argument('--restore_path', default='/data/yqwang/Project/3dDescriptor/train_mincv_cb_gi/saved_models_045_ext/training_model-99999',
                    type=str,
                    help='path to the saved model')

parser.add_argument('--gi_size', default=32, type=int,
                    help='length and width of geometry image, assuming it\'s square')
parser.add_argument('--gi_channel', default=2, type=int,
                    help='number of geometry image channels')


parser.add_argument('--gi_dir', '--gd', default='/data/yqwang/Dataset/faust_test_045_cb/',
                    type=str, help='root directory of gi files')
parser.add_argument('--desc_dir', '--dd', default='/data/yqwang/Project/3dDescriptor/evaluation/descs_99999_045_cb_gi_ext/',
                    type=str, help='directory of descriptors')

global args


# In[3]:

# In[4]:

# In[5]:

class TripletNet:
    def __init__(self, args=None, is_training=True):
        self.args = args
        self.is_training = is_training
        # self.predict_net =None
        self.anchor_net = None  # anchor_net is also the predict_net
        self.positive_net = None
        self.negative_net = None
        self.descriptors = None  # descriptors of anchors
        self.cost = None
        self.cost_same = None
        self.cost_diff = None
        self.all_multiuse_params = None
        self.predictions = None
        self.acc = None

    def inference(self, gi_placeholder, reuse=None):  # reuse=None is equal to reuse=False(i.e. don't reuse)
        with tf.variable_scope('model', reuse=reuse):
            tl.layers.set_name_reuse(reuse)  # reuse!

            network = tl.layers.InputLayer(gi_placeholder, name='input')

            """ conv2 """
            network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.identity,
                             padding='SAME', W_init=args.conv_initializer, name='conv2_1')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn2_1')

            network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                padding='SAME', name='pool2')

            """ conv3 """
            network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.identity,
                             padding='SAME', W_init=args.conv_initializer, name='conv3_1')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn3_1')

            network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                padding='SAME', name='pool3')

            """ conv4 """
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.identity,
                             padding='SAME', W_init=args.conv_initializer, name='conv4_1')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn4_1')

            network = MeanPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                 padding='SAME', name='pool4')


            network = FlattenLayer(network, name='flatten')
            network = DenseLayer(network, n_units=512, act=tf.identity, name='fc1_relu')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn_fc')
            network = DenseLayer(network, n_units=256, act=tf.identity, name='128d_embedding')

        return network


    def build_nets(self, anchor_placeholder, positive_placeholder, negative_placeholder, anchor_label_placeholder, keypoint_num=None):
        self.anchor_net = self.inference(anchor_placeholder, reuse=None)
        self.descriptors = self.anchor_net.outputs



# In[6]:
args = parser.parse_args()
# args = parser.parse_args(args=['-g 1'])
setattr(args, 'conv_initializer', tf.contrib.layers.xavier_initializer())
setattr(args, 'activation', tl.activation.leaky_relu)


os.makedirs(args.desc_dir, exist_ok=True)
required_gi_shape = (args.gi_size, args.gi_size, args.gi_channel)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)




# In[7]:

def read_gi(path) -> np.ndarray:
    # Open a gi file and return the content as numpy.ndarray

    tensor = [[]]
    with open(path, 'r') as text:
        for line in text:
            if line == '\n':
                tensor.append([])
            else:
                tensor[-1].append([float(i) for i in line.strip(' \t\n').split()])

    while not tensor[-1]:
        del tensor[-1]

    rtensor = np.asarray(tensor, dtype=np.float32).transpose((1, 2, 0))

    assert rtensor.shape == required_gi_shape, 'The dimension of gi does not match with the CNNs. \n' + \
                                               'Requires %s but receives %s' % (str(required_gi_shape), str(rtensor.shape))

    return rtensor


def write_descriptors(path, descriptor_list):
    """
    Write descriptors in a text file

    :param path: path of the target text file.
    :descriptor_list: list of either descriptor vector to be written, or an integer -1 which represent non-existence.

    """

    with open(path, 'w') as descs:
        for descriptor in descriptor_list:
            if descriptor.__class__ == int:
                descs.write(str(descriptor) + '\n')
            else:     
                desc_len = len(descriptor)
                for i, val in enumerate(descriptor):
                    if i != desc_len - 1:
                        descs.write(str(val) + ',')
                    else:
                        descs.write(str(val) + '\n')
    return


# In[8]:

run_time = time.localtime(time.time())

# [batch_size, height, width, channels]
anchor_placeholder = tf.placeholder(
    dtype=tf.float32,
    shape=[None, args.gi_size, args.gi_size, args.gi_channel])  # [batch_size, height, width, channels]

positive_placeholder = tf.placeholder(
    dtype=tf.float32,
    shape=[None, args.gi_size, args.gi_size, args.gi_channel])  # [batch_size, height, width, channels]

negative_placeholder = tf.placeholder(
    dtype=tf.float32,
    shape=[None, args.gi_size, args.gi_size, args.gi_channel])  # [batch_size, height, width, channels]

anchor_label_placeholder = tf.placeholder(
    dtype=tf.int32,
    shape=[None])  # [batch_size, height, width, channels]

triplet_net = TripletNet(is_training=False) # testing 

triplet_net.build_nets(
    anchor_placeholder=anchor_placeholder,
    positive_placeholder=positive_placeholder,
    negative_placeholder=negative_placeholder,
    anchor_label_placeholder=anchor_label_placeholder
)

train_params = triplet_net.anchor_net.all_params


# temp = set(tf.global_variables())

# if args.restore:
#     tl.layers.initialize_global_variables(sess)
#     load_saver = tf.train.Saver()
#     load_saver.restore(sess, args.restore_path)
#     info = 'Restore model parameters from %s' % args.restore_path
#     print(info)

# else:
#     tl.layers.initialize_global_variables(sess)
#     info = 'Successfully initialized global variables.'
#     print(info)       

load_saver = tf.train.Saver()
load_saver.restore(sess, args.restore_path)
info = 'Restore model parameters from %s' % args.restore_path
print(info)


# sess.run(tf.initialize_variables(set(tf.global_variables()) - temp))

    
triplet_net.anchor_net.print_params()
triplet_net.anchor_net.print_layers()


# start_time = time.time()    

model_folder_list = os.listdir(args.gi_dir)

n_models = len(model_folder_list)

for i, model_folder in enumerate(model_folder_list):

    start_time = time.time() 
    print('Processing %s, %d of %d' % (model_folder, i + 1, n_models))

    model_id = int(model_folder.split('_')[-1]) #_->.
    current_model_dir = join(args.gi_dir, model_folder)
    n_allpoints = len(os.listdir(current_model_dir))
    #current_descriptor_path = join(args.desc_dir, 'tr_reg_%.3d.desc' % model_id)
    current_descriptor_path = join(args.desc_dir, '%s.desc' % model_folder)

    gi_list = []
    # nonexist_point_id_list = []

    print('Loading gi files...')
    for point_id in tqdm(range(n_allpoints)):
        current_gi_path = join(current_model_dir, 'tr_reg_%.3d_pidx_%.4d.gi' % (model_id, point_id))
        #current_gi_path = join(current_model_dir, '%s_pidx_%.4d.gi' % (model_folder, point_id))
        assert os.path.isfile(current_gi_path), 'One required gi file not found! Filepath: %s' % current_gi_path
        gi_list.append(read_gi(current_gi_path))
        # if os.path.isfile(current_gi_path):
        #     gi_list.append(read_gi(current_gi_path))
        # else:
            # nonexist_point_id_list.append(point_id)
    gi_list_batch_list = (gi_list[i: i + args.batch_size] for i in range(0, len(gi_list), args.batch_size))

    print('Generating descriptors...')
    current_model_desc_list = []

    for gi_list_batch in gi_list_batch_list:
        current_model_desc_list.extend(list(sess.run(triplet_net.descriptors,
                                                     feed_dict={anchor_placeholder: np.asarray(gi_list_batch)})))
    # current_model_desc_list = list(current_model_descriptors)

    # for nonexist_point_id in nonexist_point_id_list:
    #     current_model_desc_list.insert(nonexist_point_id, -1)  # Value -1 means the descriptor of this point does not exist.

    write_descriptors(current_descriptor_path, current_model_desc_list)

    print('Descriptors of %s generated, time cost: %fs' % (model_folder, time.time() - start_time))

