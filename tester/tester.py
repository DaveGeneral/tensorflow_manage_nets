import time
import tensorflow as tf
import sys, os
import numpy as np

switch_server = True

testdir = os.path.dirname('__file__')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

if switch_server is True:
    from tools import utils
    from nets import net_alex2 as ALEX
    from tools.dataset_image import Dataset
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.nets import net_alex2 as ALEX
    from tensorflow_manage_nets.tools.dataset_image import Dataset



# GLOBAL VARIABLES
path = '../../data/CIFAR_10/'
path_dir_image_train = path + "train_cifar10_original/"
path_dir_image_test = path + "test_cifar10_original/"
path_data_train = path + 'cifar10_train_label.csv'
path_data_test = path + 'cifar10_test_label.csv'

if __name__ == '__main__':

    # LOad y save  weights
    path_load_weight = '../weight/alexnet.npy'
    path_save_weight = '../weight/save_alexnet.npy'
    load_weight_fc = False

    # Ultimas capas de la red
    # last_layers = [128, 10]
    num_class = 10
    epoch = 4
    mini_batch_train = 20
    mini_batch_test = 30
    learning_rate = 0.0001
    accuracy = 0

    with tf.Session() as sess:
        # DEFINE MODEL
        vgg_batch = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg_label = tf.placeholder(tf.float32, [None, num_class])
        train_mode = tf.placeholder(tf.bool)

        # Initialize of the model VGG19
        vgg = ALEX.ALEXNET(path_load_weight, learning_rate=learning_rate, load_weight_fc=load_weight_fc)
        vgg.build(vgg_batch, vgg_label, train_mode, num_class=num_class)
        sess.run(tf.global_variables_initializer())
