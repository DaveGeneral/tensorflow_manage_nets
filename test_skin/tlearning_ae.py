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
    from nets import net_aencoder as AE
    from tools.dataset_csv import Dataset_csv
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.nets import net_aencoder as AE
    from tensorflow_manage_nets.tools.dataset_csv import Dataset_csv

# ..................................................................


# Función, fase de test
def test_model(net, sess_test, objData):

    total = objData.total_inputs
    cost_total = 0

    for i in range(objData.total_batchs_complete):

        x_, label = objData.generate_batch()
        mask_np = np.random.binomial(1, 1 - net.noise, x_.shape)
        cost = sess_test.run(net.cost, feed_dict={x_batch: x_, mask: mask_np, noise_mode: False})

        cost_total = cost_total + cost
        objData.next_batch_test()

    return cost_total, cost_total/total


# Funcion, fase de entrenamiento
def train_model(net, sess_train, objData, objDatatest, epoch):

    print('\n# PHASE: Training model')
    for ep in range(epoch):
        for i in range(objData.total_batchs):

            batch, _ = objData.generate_batch()
            mask_np = np.random.binomial(1, 1 - net.noise, batch.shape)
            sess_train.run(net.train, feed_dict={x_batch: batch, mask: mask_np, noise_mode: False})
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
        mask_np = np.random.binomial(1, 1 - net0.noise, x_.shape)
        cost0 = sess_test.run(net0.cost, feed_dict={x_batch: x_, mask: mask_np, noise_mode: False})
        cost1 = sess_test.run(net1.cost, feed_dict={x_batch: x_, mask: mask_np, noise_mode: False})

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
    mask_np = np.random.binomial(1, 1 - net.noise, batch.shape)
    recon = sess.run(net.y, feed_dict={x_batch: batch, mask: mask_np, noise_mode: False})
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
path_save_weight_dual = '../weight/tlaencode_dual_'+OPC+'_1.npy'
path_load_weight0 = None
path_save_weight0 = '../weight/tlaencode_class0_'+OPC+'_1.npy'
path_load_weight1 = None
path_save_weight1 = '../weight/tlaencode_class1_'+OPC+'_1.npy'


if __name__ == '__main__':

    mini_batch_train = 20
    mini_batch_test = 30
    epoch = 10
    learning_rate = 0.00001
    noise_level = 0

    # Datos de valor maximo
    data_normal = Dataset_csv(path_data=[path_data_train_dual[0], path_data_test_dual[0]], random=False)
    Damax = data_normal.amax
    del data_normal

    # -------------------------------------------------------------------
    # ENTRENAMOS EL AUTOENCODER CON AMBAS CLASES - GENERAMOS UN PESO BASE
    # -------------------------------------------------------------------
    print('AE TRAIN DUAL')
    print('-------------')

    data_train = Dataset_csv(path_data=path_data_train_dual, minibatch=mini_batch_train, max_value=Damax)
    data_test = Dataset_csv(path_data=path_data_test_dual, minibatch=mini_batch_test, max_value=Damax, random=False)

    with tf.Session() as sess:

        x_batch = tf.placeholder(tf.float32, [None, 4096])
        mask = tf.placeholder(tf.float32, [None, 4096])
        noise_mode = tf.placeholder(tf.bool)

        AEncode = AE.AEncoder(path_load_weight_dual, learning_rate=learning_rate, noise=noise_level)
        AEncode.build(x_batch, mask, noise_mode, [2048, 1024])
        sess.run(tf.global_variables_initializer())

        print('Original Cost: ', test_model(AEncode, sess, data_test))
        train_model(AEncode, sess, data_train, objDatatest=data_test, epoch=epoch)

        # SAVE WEIGHTs
        AEncode.save_npy(sess, path_save_weight_dual)

        # Plot example reconstructions
        # plot_result(AEncode, sess, data_train.generate_batch()[0])

    del AEncode
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
        mask = tf.placeholder(tf.float32, [None, 4096])
        noise_mode = tf.placeholder(tf.bool)

        # Default: path_save_weight_dual
        AEncode = AE.AEncoder(path_save_weight_dual, learning_rate=learning_rate, noise=noise_level)
        AEncode.build(x_batch, mask, noise_mode, [2048, 1024])
        sess.run(tf.global_variables_initializer())

        print('Original Cost: ', test_model(AEncode, sess, data_test))
        train_model(AEncode, sess, data_train, objDatatest=data_test, epoch=epoch)

        # SAVE WEIGHTs
        AEncode.save_npy(sess, path_save_weight0)

        # Plot example reconstructions
        # plot_result(AEncode, sess, data_train.generate_batch()[0])

    del AEncode
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
        mask = tf.placeholder(tf.float32, [None, 4096])
        noise_mode = tf.placeholder(tf.bool)

        # Default: path_save_weight_dual
        AEncode = AE.AEncoder(path_save_weight_dual, learning_rate=learning_rate, noise=noise_level)
        AEncode.build(x_batch, mask, noise_mode, [2048, 1024])
        sess.run(tf.global_variables_initializer())

        print('Original Cost: ', test_model(AEncode, sess, data_test))
        train_model(AEncode, sess, data_train, objDatatest=data_test, epoch=epoch)

        # SAVE WEIGHTs
        AEncode.save_npy(sess, path_save_weight1)

        # Plot example reconstructions
        # plot_result(AEncode, sess, data_train.generate_batch()[0])

    del AEncode
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
        mask = tf.placeholder(tf.float32, [None, 4096])
        noise_mode = tf.placeholder(tf.bool)

        AEBenigno = AE.AEncoder(path_save_weight0, learning_rate=learning_rate, noise=noise_level)
        AEBenigno.build(x_batch, mask, noise_mode, [2048, 1024])
        AEMaligno = AE.AEncoder(path_save_weight1, learning_rate=learning_rate, noise=noise_level)
        AEMaligno.build(x_batch, mask, noise_mode, [2048, 1024])

        sess.run(tf.global_variables_initializer())
        test_model_dual(AEBenigno, AEMaligno, sess, data_test_dual)

