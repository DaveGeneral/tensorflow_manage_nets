import time
import tensorflow as tf
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

switch_server = True

testdir = os.path.dirname('__file__')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

if switch_server is True:
    from tools import utils
    from nets import net_conv_aencoder as CAE
    from tools.dataset_csv import Dataset_csv
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.nets import net_conv_aencoder as CAE
    from tensorflow_manage_nets.tools.dataset_csv import Dataset_csv

# ..................................................................


# Función, fase de test
def test_model(net, sess_test, objData):

    total = objData.total_inputs
    cost_total = 0

    for i in range(objData.total_batchs_complete):

        x_, label = objData.generate_batch()
        cost = sess_test.run(net.cost, feed_dict={x_batch: x_})

        cost_total = cost_total + cost
        objData.next_batch_test()

    return cost_total, cost_total/total


# Funcion, fase de entrenamiento
def train_model(net, sess_train, objData, objDatatest, epoch):

    print('\n# PHASE: Training model')
    for ep in range(epoch):
        for i in range(objData.total_batchs):

            batch, _ = objData.generate_batch()
            sess_train.run(net.train, feed_dict={x_batch: batch})
            objData.next_batch()

        cost_tot, cost_prom = test_model(net, sess_train, objDatatest)
        print('     Epoch', ep, ': ', cost_tot, ' / ', cost_prom)


# Función, Fase de test - clasificacion
def test_model_dual(net0, net1, sess_test, objData):

    total = objData.total_inputs
    count_accu = 0

    res = []
    for i in range(objData.total_batchs_complete):

        x_, label = objData.generate_batch()
        cost0 = sess_test.run(net0.cost, feed_dict={x_batch: x_})
        cost1 = sess_test.run(net1.cost, feed_dict={x_batch: x_})

        if cost0 < cost1:
            cen = 0
        else:
            cen = 1

        res.append(cen)
        if label[0] == cen:
            count_accu = count_accu + 1

        objData.next_batch_test()

    print(total)
    print(count_accu)
    print('AC: ', count_accu / total)

    a = objData.labels
    b = res
    cm = confusion_matrix(a, b)
    print(cm)


# Plot example reconstructions
def plot_result(net, sess, batch, n_examples=5):
    print('Plotting')
    recon = sess.run(net.y, feed_dict={x_batch: batch})
    fig, axs = plt.subplots(2, n_examples, figsize=(n_examples, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(np.reshape(batch[example_i, :], (64, 64)))
        axs[1][example_i].imshow(np.reshape(recon[example_i, :], (64, 64)))

    fig.show()
    plt.draw()
    plt.waitforbuttonpress()

# ..................................................................
#     Opcion para ejecutar la red con distintas fuentes de datos,
#                   son tres fuentes de datos

OPC = 'A'
# ..................................................................

# GLOBAL VARIABLES
path = '../data/features/muestra'+OPC+'/'
path_data_train0 = [path + 'SKINfeatures'+OPC+'_Train_class0.csv']
path_data_train1 = [path + 'SKINfeatures'+OPC+'_Train_class1.csv']
path_data_test0 = [path + 'SKINfeatures'+OPC+'_Test_class0.csv']
path_data_test1 = [path + 'SKINfeatures'+OPC+'_Test_class1.csv']

path_data_train_dual = [path + 'SKINfeatures'+OPC+'_Train.csv']
path_data_test_dual = [path + 'SKINfeatures'+OPC+'_Test.csv']

path_load_weight_dual = None
path_save_weight_dual = '../weight/tlconvolae_dual_'+OPC+'_1.npy'

path_load_weight0 = '../weight/tlconvolae_dual_'+OPC+'_1.npy'
path_save_weight0 = '../weight/tlconvolae_class0_'+OPC+'_1.npy'
path_load_weight1 = '../weight/tlconvolae_dual_'+OPC+'_1.npy'
path_save_weight1 = '../weight/tlconvolae_class1_'+OPC+'_1.npy'


if __name__ == '__main__':

    mini_batch_train = 34
    mini_batch_test = 30
    epoch = 10
    learning_rate = 0.0001

    # Datos de valor maximo
    data_normal = Dataset_csv(path_data=[path_data_train_dual[0], path_data_test_dual[0]], random=False)
    Damax = data_normal.amax
    del data_normal

    # -------------------------------------------------------------------
    # ENTRENAMOS EL AUTOENCODER CON AMBAS CLASES - GENERAMOS UN PESO BASE
    # -------------------------------------------------------------------
    print('ConvAE TRAIN DUAL')
    print('-----------------')

    data_train = Dataset_csv(path_data=path_data_train_dual, minibatch=mini_batch_train, max_value=Damax)
    data_test = Dataset_csv(path_data=path_data_test_dual, minibatch=mini_batch_test, max_value=Damax, random=False)

    with tf.Session() as sess:

        x_batch = tf.placeholder(tf.float32, [None, 4096])

        CAEncode = CAE.ConvAEncoder(path_load_weight_dual, learning_rate=learning_rate)
        CAEncode.build(input_batch=x_batch, n_filters=[1, 10, 10], corruption=False)
        sess.run(tf.global_variables_initializer())

        print('Original Cost: ', test_model(CAEncode, sess, data_test))
        train_model(CAEncode, sess, data_train, objDatatest=data_test, epoch=epoch)

        # SAVE WEIGHTs
        CAEncode.save_npy(sess, path_save_weight_dual)

        # Plot example reconstructions
        # plot_result(AEncode, sess, data_train.generate_batch()[0])

    del CAEncode
    del data_train
    del data_test

    # -------------------------------------------------------------------
    #       ENTRENAMOS EL AUTOENCODER CON LA CLASE 0 - BENIGNO
    # -------------------------------------------------------------------
    print()
    print('AE TRAIN CLASS 0')
    print('----------------')
    epoch = 10

    data_train = Dataset_csv(path_data=path_data_train0, minibatch=mini_batch_train, max_value=Damax)
    data_test = Dataset_csv(path_data=path_data_test0, minibatch=mini_batch_test, max_value=Damax, restrict=False, random=False)

    with tf.Session() as sess:

        x_batch = tf.placeholder(tf.float32, [None, 4096])

        # Default: path_save_weight_dual
        CAEncode = CAE.ConvAEncoder(path_load_weight0, learning_rate=learning_rate)
        CAEncode.build(input_batch=x_batch, n_filters=[1, 10, 10], corruption=False)
        sess.run(tf.global_variables_initializer())

        print('Original Cost: ', test_model(CAEncode, sess, data_test))
        train_model(CAEncode, sess, data_train, objDatatest=data_test, epoch=epoch)

        # SAVE WEIGHTs
        CAEncode.save_npy(sess, path_save_weight0)

        # Plot example reconstructions
        # plot_result(AEncode, sess, data_train.generate_batch()[0])

    del CAEncode
    del data_train
    del data_test

    # -------------------------------------------------------------------
    #       ENTRENAMOS EL AUTOENCODER CON LA CLASE 1 - MALIGNO
    # -------------------------------------------------------------------
    print()
    print('AE TRAIN CLASS 1')
    print('----------------')
    epoch = 10

    data_train = Dataset_csv(path_data=path_data_train1, minibatch=mini_batch_train, max_value=Damax)
    data_test = Dataset_csv(path_data=path_data_test1, minibatch=mini_batch_test, max_value=Damax, restrict=False,
                            random=False)

    with tf.Session() as sess:
        x_batch = tf.placeholder(tf.float32, [None, 4096])

        # Default: path_save_weight_dual
        CAEncode = CAE.ConvAEncoder(path_load_weight1, learning_rate=learning_rate)
        CAEncode.build(input_batch=x_batch, n_filters=[1, 10, 10], corruption=False)
        sess.run(tf.global_variables_initializer())

        print('Original Cost: ', test_model(CAEncode, sess, data_test))
        train_model(CAEncode, sess, data_train, objDatatest=data_test, epoch=epoch)

        # SAVE WEIGHTs
        CAEncode.save_npy(sess, path_save_weight1)

        # Plot example reconstructions
        # plot_result(AEncode, sess, data_train.generate_batch()[0])

    del CAEncode
    del data_train
    del data_test

    # -------------------------------------------------------------------
    #                   CLASIFICACION CON AUTOENCODERS
    # -------------------------------------------------------------------
    print()
    print('AE CLASSIFICATION')
    print('-----------------')

    data_test_dual = Dataset_csv(path_data=path_data_test_dual, minibatch=1, max_value=Damax, restrict=False, random=False)

    with tf.Session() as sess:
        x_batch = tf.placeholder(tf.float32, [None, 4096])

        CAEBenigno = CAE.ConvAEncoder(path_save_weight0, learning_rate=learning_rate)
        CAEBenigno.build(input_batch=x_batch, n_filters=[1, 10, 10], corruption=False)
        CAEMaligno = CAE.ConvAEncoder(path_save_weight1, learning_rate=learning_rate)
        CAEMaligno.build(input_batch=x_batch, n_filters=[1, 10, 10], corruption=False)

        sess.run(tf.global_variables_initializer())
        test_model_dual(CAEBenigno, CAEMaligno, sess, data_test_dual)

