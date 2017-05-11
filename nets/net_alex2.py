import os
import tensorflow as tf
import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]


class ALEXNET:

    def __init__(self, npy_path=None, trainable=True, learning_rate=0.05, dropout=0.5, load_weight_fc=False):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
            print("npy file loaded")
        else:
            self.data_dict = None
            print("random weight")

        print(self.data_dict['fc7'])
        self.var_dict = {}
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.load_weight_fc = load_weight_fc

    def build(self, rgb, target, train_mode=True, last_layers=[4096, 1000]):
        """
        load variable from npy to build the vgg

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param target: label image [#clases]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        :param size_layer_fc: size of the last layer classification by default vgg19 use 4096
        """
        self.num_class = last_layers[-1]

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

        self.conv1 = self.conv_layer(bgr, 3, 96, 11, 4, 4, padding='VALID', name="conv1")
        self.lrn1 = tf.nn.lrn(self.conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='norm1')
        self.pool1 = self.max_pool(self.lrn1, 3, 3, 2, 2, padding='VALID', name='pool1')

        self.conv2 = self.conv_layer(self.pool1, 96, 256, 5, 1, 1, group=2, name="conv2")
        self.lrn2 = tf.nn.lrn(self.conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='norm2')

        self.pool2 = self.max_pool(self.lrn2, 3, 3, 2, 2, padding='VALID', name='pool1')

        self.conv3 = self.conv_layer(self.pool2, 256, 384, 3, 1, 1, name="conv3")
        self.conv4 = self.conv_layer(self.conv3, 384, 384, 3, 1, 1, group=2, name="conv4")
        self.conv5 = self.conv_layer(self.conv4, 384, 256, 3, 1, 1, group=2, name="conv5")
        self.pool5 = self.max_pool(self.conv5, 3, 3, 2, 2, padding='SAME', name='pool5')

        self.fc6 = self.fc_layer(self.pool5, 9216, 4096, "fc6", load_weight_force=True) # 9216 = 256 * 6 * 6
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer_sigmoid(self.relu6, 4096, last_layers[0], "fc7", load_weight_force=True)

        self.fc8 = self.fc_layer(self.relu7, last_layers[0], last_layers[1], "fc8", load_weight_force=False)
        self.prob = tf.nn.softmax(self.fc8, name="prob")

        # COST - TRAINING
        self.cost = tf.reduce_mean((self.prob - target) ** 2)
        # self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    # Layer MaxPool
    def max_pool(self, bottom, k_h, k_w, s_h, s_w, padding='SAME', name=''):
        return tf.nn.max_pool(bottom, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name)

    # Layer Convolutional
    def conv_layer(self, bottom, in_channels, out_channels, filter_size=3, s_h=1, s_w=1, group=1, padding='SAME', name=''):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, int(in_channels / group), out_channels, name)

            if group == 1:
                conv = tf.nn.conv2d(bottom, filt, [1, s_h, s_w, 1], padding=padding)
                bias = tf.nn.bias_add(conv, conv_biases)
                relu = tf.nn.relu(bias)
            else:
                input_groups = tf.split(bottom, group, axis=3)
                kernel_groups = tf.split(filt, group, axis=3)
                conv_groups = []
                for i in range(group):
                    conv_groups.append(tf.nn.conv2d(input_groups[i], kernel_groups[i], [1, s_h, s_w, 1], padding=padding))

                # Concatenate the groups
                output_conv = tf.concat(conv_groups, axis=3)
                bias = tf.nn.bias_add(output_conv, conv_biases)
                relu = tf.nn.relu(bias)
            return relu

    # Generate Parameter Convol layer
    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var_conv(initial_value, name, 'weights', name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var_conv(initial_value, name, 'biases', name + "_biases")

        return filters, biases

    # Construct dictionary with random parameters or load parameters
    def get_var_conv(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            print(name, idx)
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

    # Layer Sigmoid
    def fc_layer_sigmoid(self, bottom, in_size, out_size, name, load_weight_force=False):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, load_weight_force)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.sigmoid(tf.matmul(x, weights) + biases)
            return fc

    # Generate Parameter FullConnect layer
    def get_fc_var(self, in_size, out_size, name, load_wf=False):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var_fc(initial_value, name, 'weights', name + "_weights", load_wf)

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var_fc(initial_value, name, 'biases', name + "_biases", load_wf)

        return weights, biases

    # Construct dictionary with random parameters or load parameters
    def get_var_fc(self, initial_value, name, idx, var_name, load_wf=False):
        if self.data_dict is not None and name in self.data_dict and ((self.load_weight_fc is True) or (load_wf is True)):
            print(name, idx)
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        # print(var.get_shape(), initial_value.get_shape())
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
