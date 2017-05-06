import os
import tensorflow as tf
import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]


class VGG19:

    def __init__(self, npy_path=None, trainable=True, learning_rate=0.05, dropout=0.5, load_weight_fc=False):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
            print("npy file loaded")
        else:
            self.data_dict = None
            print("random weight")

        self.var_dict = {}
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.load_weight_fc = load_weight_fc

    def build(self, rgb, target, train_mode=True, size_layer_fc=4096, num_class=1000):
        """
        load variable from npy to build the vgg

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param target: label image [#clases]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        :param size_layer_fc: size of the last layer classification by default vgg19 use 4096
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6", load_weight_force=True) # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)

        # DROPOUT
        if self.trainable is True:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)

        self.fc7 = self.fc_layer(self.relu6, 4096, size_layer_fc, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        # DROPOUT
        if self.trainable is True:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)

        self.fc8 = self.fc_layer(self.relu7, size_layer_fc, num_class, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")

        # COST - TRAINING
        self.cost = tf.reduce_mean((self.prob - target) ** 2)
        # self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    # Layer MaxPool
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # Layer Convolutional
    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    # Generate Parameter Convol layer
    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var_conv(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var_conv(initial_value, name, 1, name + "_biases")

        return filters, biases

    # Construct dictionary with random parameters or load parameters
    def get_var_conv(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()
        return var

    # Layer FullConnected
    def fc_layer(self, bottom, in_size, out_size, name, load_weight_force=False):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, load_weight_force)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    # Generate Parameter FullConnect layer
    def get_fc_var(self, in_size, out_size, name, load_wf=False):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var_fc(initial_value, name, 0, name + "_weights", load_wf)

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var_fc(initial_value, name, 1, name + "_biases", load_wf)

        return weights, biases

    # Construct dictionary with random parameters or load parameters
    def get_var_fc(self, initial_value, name, idx, var_name, load_wf=False):
        if self.data_dict is not None and name in self.data_dict and ((self.load_weight_fc is True) or (load_wf is True)):
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()
        return var

    # Save weight model
    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("File saved", npy_path)
        return npy_path
