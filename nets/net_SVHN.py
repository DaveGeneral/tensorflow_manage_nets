import os
import tensorflow as tf
import numpy as np
import time
import inspect

#VGG_MEAN = [103.939, 116.779, 123.68]

class SVHN_NET:

    def __init__(self, npy_path=None, trainable=True, l_rate=0.075, decay_rate=0.95, dropout=0.85,
                 num_class=10, minibatch=256, train_size=0, test_size=0):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
            print("npy file loaded")
        else:
            self.data_dict = None
            print("random weight")

        self.var_dict = {}
        self.trainable = trainable
        self.l_rate = l_rate
        self.decay_rate = decay_rate
        self.dropout = dropout
        self.num_class = num_class
        self.minibatch = minibatch
        self.train_size = train_size
        self.test_size = test_size

    def build(self, input_batch, input_label, train_mode=True, load_lrate=False):

        if self.data_dict is not None and 'l_rate' in self.data_dict and load_lrate is True:
            value = self.data_dict['l_rate'][0]
            self.l_rate = value
            print('Load Learning_rate...')

        global_step = tf.Variable(0, trainable=False)

        start_time = time.time()
        print("build model started")

        self.conv1 = self.conv_layer(input_batch, 3, 48, 5, 1, 1, padding='VALID', relu=True, name="conv1", bias=0.0)
        self.pool1 = self.max_pool(self.conv1, 2, 2, 2, 2, padding='SAME', name='pool1')
        self.lrn1 = tf.nn.local_response_normalization(self.pool1, name='norm1')

        self.conv2 = self.conv_layer(self.lrn1, 48, 64, 5, 1, 1, padding='VALID', relu=True, name="conv2", bias=0.1)
        self.pool2 = self.max_pool(self.conv2, 2, 2, 2, 2, padding='SAME', name='pool2')
        self.lrn2 = tf.nn.local_response_normalization(self.pool2, name='norm2')

        self.conv3 = self.conv_layer(self.lrn2, 64, 128, 5, 1, 1, padding='SAME', relu=True, name="conv3", bias=0.1)
        self.lrn3 = tf.nn.local_response_normalization(self.conv3, name='norm3')

        if self.lrn3.get_shape().as_list()[1] is 1:  # Is already reduced.
            self.pool3 = self.max_pool(self.lrn3, 1, 1, 1, 1, padding='SAME', name='pool3')
        else:
            self.pool3 = self.max_pool(self.lrn3, 2, 2, 2, 2, padding='SAME', name='pool3')

        shape = self.pool3.get_shape().as_list()

        self.pool3 = tf.cond(train_mode, lambda: tf.nn.dropout(self.pool3, self.dropout), lambda: self.pool3)
        #self.fc1 = tf.reshape(self.pool3, [shape[0], -1])

        self.fc1 = self.fc_layer(self.pool3, 1152, 160, "fc1", bias=0.05)
        self.relu1 = tf.nn.relu(self.fc1)

        self.logits = self.fc_layer(self.relu1, 160, 10, "fc2", bias=0.05)

        self.prob = tf.nn.softmax(self.logits)
        self.prediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(input_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=input_label))
        self.learning_rate = tf.train.exponential_decay(self.l_rate, global_step * self.minibatch, self.train_size, self.decay_rate, staircase=True)
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
        #self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        #self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    # Layer MaxPool
    def max_pool(self, bottom, k_h, k_w, s_h, s_w, padding='SAME', name=''):
        #print(name,np.shape(bottom))
        return tf.nn.max_pool(bottom, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name)

    # Layer Convolutional
    def conv_layer(self, bottom, in_channels, out_channels, filter_size=3, s_h=1, s_w=1, group=1, padding='SAME', relu=True, name='', bias=0.0):
        #print(name,np.shape(bottom))
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, int(in_channels / group), out_channels, name, bias)

            if group == 1:
                conv = tf.nn.conv2d(bottom, filt, [1, s_h, s_w, 1], padding=padding)
                conv = tf.nn.bias_add(conv, conv_biases)
                if relu is True:
                    conv = tf.nn.relu(conv)
            else:
                input_groups = tf.split(bottom, group, axis=3)
                kernel_groups = tf.split(filt, group, axis=3)
                conv_groups = []
                for i in range(group):
                    conv_groups.append(tf.nn.conv2d(input_groups[i], kernel_groups[i], [1, s_h, s_w, 1], padding=padding))

                # Concatenate the groups
                output_conv = tf.concat(conv_groups, axis=3)
                conv = tf.nn.bias_add(output_conv, conv_biases)
                if relu is True:
                    conv = tf.nn.relu(conv)

            return conv

    # Generate Parameter Convol layer
    def get_conv_var(self, filter_size, in_channels, out_channels, name, bias=0.0):
        #initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        initial_value = tf.get_variable(name + "_filters", shape=[filter_size, filter_size, in_channels, out_channels])
        filters = self.get_var_conv(initial_value, name, 'weights', name + "_filters")

        #initial_value = tf.truncated_normal([out_channels], .0, .001)
        initial_value = tf.constant(bias, shape=[out_channels])
        biases = self.get_var_conv(initial_value, name, 'biases', name + "_biases")

        return filters, biases

    # Construct dictionary with random parameters or load parameters
    def get_var_conv(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            print(name, idx)
            value = self.data_dict[name][idx]
            var = tf.Variable(value, name=var_name)
        else:
            value = initial_value
            if idx == 'biases':
                var = tf.Variable(value, name=var_name)
            else:
                var = value

        #print(var.get_shape(), initial_value.get_shape())
        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()
        return var

    # Layer FullConnected
    def fc_layer(self, bottom, in_size, out_size, name, load_weight_force=True, bias=0.0):
        #print(name,np.shape(bottom))
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, load_weight_force, bias)
            #x = bottom
            x = tf.reshape(bottom, [-1, in_size])
            #x = tf.reshape(bottom, [shape[0], -1])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    # Layer Sigmoid
    def fc_layer_sigmoid(self, bottom, in_size, out_size, name, load_weight_force=True):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, load_weight_force)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.sigmoid(tf.matmul(x, weights) + biases)
            return fc

    # Generate Parameter FullConnect layer
    def get_fc_var(self, in_size, out_size, name, load_wf=True, bias=0.0):
        # initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        initial_value = tf.get_variable(name + "_weights", shape=[in_size, out_size])
        weights = self.get_var_fc(initial_value, name, 'weights', name + "_weights", load_wf)

        #initial_value = tf.truncated_normal([out_size], .0, .001)
        initial_value = tf.constant(bias, shape=[out_size])
        biases = self.get_var_fc(initial_value, name, 'biases', name + "_biases", load_wf)

        return weights, biases

    # Construct dictionary with random parameters or load parameters
    def get_var_fc(self, initial_value, name, idx, var_name, load_wf=True):
        if self.data_dict is not None and name in self.data_dict and (load_wf is True):
            print(name, idx)
            value = self.data_dict[name][idx]
            var = tf.Variable(value, name=var_name)
        else:
            value = initial_value
            if idx == 'biases':
                var = tf.Variable(value, name=var_name)
            else:
                var = value

        #print(var.get_shape(), initial_value.get_shape())
        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()
        return var

    # Save weight model
    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)
        self.var_dict[('l_rate', 0)] = self.learning_rate
        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("File saved", npy_path)
        return npy_path

