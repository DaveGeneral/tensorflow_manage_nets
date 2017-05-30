import time
import tensorflow as tf
import sys, os
import numpy as np
from datetime import datetime

switch_server = True

testdir = os.path.dirname('__file__')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

if switch_server is True:
    from tools import utils
    from nets import net_SVHN as SVHN
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.nets import net_SVHN as SVHN


# GLOBAL VARIABLES


# LECTURA DE DATOS
def load_svhn_data(data_type, data_set_name):
    # TODO add error handling here
    path = "../../data/SVHN/" + data_set_name
    imgs = np.load(os.path.join(path, data_set_name+'_'+data_type+'_imgs.npy'))
    labels = np.load(os.path.join(path, data_set_name+'_'+data_type+'_labels.npy'))
    return imgs, labels


def fill_feed_dict(data, labels, step, batch_size):
    size = labels.shape[0]
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    offset = (step * batch_size) % (size - batch_size)
    batch_data = data[offset:(offset + batch_size), ...]
    batch_labels = labels[offset:(offset + batch_size)]
    #print(batch_labels,offset,offset+batch_size)
    return batch_data, batch_labels


def test_model(net, sess_test, objData, objLabel, xtime):

    total_acc = 0
    total_itera = int(net.test_size) // net.minibatch
    for step in range(total_itera):
        x, y = fill_feed_dict(objData, objLabel, step, net.minibatch)

        accuracy = sess_test.run(net.accuracy, feed_dict={svhn_batch: x, svhn_label: y, train_mode: False})
        total_acc = total_acc + accuracy

    acc = total_acc / total_itera
    print('Test Accuracy: %.5f%%' % acc)


# Funcion, fase de entrenamiento
def train_model(net, sess_train, objData, objLabel, epoch, xtime):

    start_time = xtime
    total_itera = int(net.train_size) // net.minibatch
    for ep in range(epoch):
        for step in range(total_itera):
            x, y = fill_feed_dict(objData, objLabel, step, net.minibatch)
            _, l, lr, acc = sess_train.run([net.optimizer, net.loss, net.learning_rate, net.accuracy], feed_dict={svhn_batch: x, svhn_label: y, train_mode: True})
            duration = time.time() - xtime

            if step % 100 == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                examples_per_sec = net.minibatch / duration
                format_str = ('%s: step %d, loss = %.2f  learning rate = %.6f  (%.1f examples/sec; %.2f ''sec/batch)')
                print(format_str % (datetime.now(), step, l, lr, examples_per_sec, duration))
                print('Mini-Batch Accuracy: %.2f%%' % acc)


if __name__ == '__main__':
    # LOad y save  weights
    path_load_weight = None
    path_save_weight = '../weight/save_SVHN.npy'

    # Ultimas capas de la red
    num_class = 10
    epoch = 2
    learning_rate = 0.075
    accuracy = 0

    # GENERATE DATA
    data_train, labels_train = load_svhn_data("train", "cropped")
    data_test, labels_test = load_svhn_data("test", "cropped")
    train_size = labels_train.shape[0]
    test_size = labels_test.shape[0]


    #c = tf.ConfigProto(log_device_placement=False)
    #c = tf.ConfigProto()
    #c.gpu_options.visible_device_list = "2"

    start_time = time.time()
    with tf.Session() as sess:
        # DEFINE MODEL
        svhn_batch = tf.placeholder(tf.float32, [None, 32, 32, 3])
        svhn_label = tf.placeholder(tf.float32, [None, num_class])
        train_mode = tf.placeholder(tf.bool)

        # Initialize of the model VGG19
        svhn = SVHN.SVHN_NET(path_load_weight, num_class=num_class, l_rate=0.075, minibatch=256, train_size=train_size, test_size=test_size)
        svhn.build(svhn_batch, svhn_label, train_mode)
        sess.run(tf.global_variables_initializer())

        # # Execute Network
        test_model(net=svhn, sess_test=sess, objData=data_test, objLabel=labels_test, xtime=start_time)
        train_model(net=svhn, sess_train=sess, objData=data_train, objLabel=labels_train, epoch=epoch, xtime=start_time)
        test_model(net=svhn, sess_test=sess, objData=data_test, objLabel=labels_test, xtime=start_time)

        svhn.save_npy(sess,path_save_weight)
