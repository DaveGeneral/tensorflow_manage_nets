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
def test_model(net, sess_test, objData, index=0):

    total = objData.total_inputs
    cost_total = 0

    for i in range(objData.total_batchs_complete):

        x_, label = objData.generate_batch()
        cost, layer = sess_test.run([net.cost, net.net['encodeFC_1']], feed_dict={x_batch: x_})

        # utils.save_layer_output(layer, label, name='endoce_cifar10_256_class'+str(index), dir='../data/features_cifar10_vgg/')
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
def test_model_all(net, sess_test, objData, numClass):

    y_true = objData.labels
    y_result = []

    for ix in range(objData.total_batchs_complete):

        x_, label = objData.generate_batch()
        cost_class = []
        for class_i in range(numClass):

            cost_i = sess_test.run(net[class_i].cost, feed_dict={x_batch: x_})
            cost_class.append(cost_i)

        y_result.append(np.argsort(cost_class)[0])
        objData.next_batch_test()

    utils.metrics_multiclass(y_true, y_result)


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


num_class = 10
capa = 'relu6'
# ..................................................................

# GLOBAL VARIABLES
path = '../data/features_cifar10_vgg/'
path_data_train_all = [path + 'output_trainVGG_'+capa+'.csv']
path_data_test_all = [path + 'output_testVGG_'+capa+'.csv']

path_weight = '../weight/vgg_cifar10/'
path_load_weight_all = None
path_save_weight_all = path_weight + 'vggAE_all.npy'

path_data_train_class = []
path_data_test_class = []
path_load_weight = []
path_save_weight = []

for i in range(num_class):
    path_data_test_class.append([path + 'output_testVGG_'+capa+'_class'+str(i)+'.csv'])
    path_data_train_class.append([path + 'output_trainVGG_'+capa+'_class' + str(i) + '.csv'])
    path_load_weight.append(path_weight + 'vggAE_all.npy')
    path_save_weight.append(path_weight + 'vggAE_class'+str(i)+'.npy')


# path = '../data/features/muestraA/'
# path_data_train_all = [path + 'SKINfeaturesA_Train.csv']
# path_data_test_all = [path + 'SKINfeaturesA_Test.csv']
#
# path_weight = '../weight/vgg_cifar10/'
# path_load_weight_all = None
# path_save_weight_all = path_weight + 'vggAE_all.npy'
#
# path_data_train_class = []
# path_data_test_class = []
# path_load_weight = []
# path_save_weight = []

# for i in range(num_class):
#     path_data_test_class.append([path + 'SKINfeaturesA_Test_class'+str(i)+'.csv'])
#     path_data_train_class.append([path + 'SKINfeaturesA_Train_class' + str(i) + '.csv'])
#     path_load_weight.append(path_weight + 'vggAE_all.npy')
#     path_save_weight.append(path_weight + 'vggAE_class'+str(i)+'.npy')


assert os.path.exists(path), print('No existe el directorio de datos ' + path)
assert os.path.exists(path_weight), print('No existe el directorio de pesos ' + path_weight)

if __name__ == '__main__':

    mini_batch_train = 25
    mini_batch_test = 30
    epoch_all = 3
    learning_rate_all = 0.0000001
    epoch_class = 10
    learning_rate_class = 0.00001

    dim_input = 4096
    layers = [[1024,'relu'], [256,'relu']]

    # Datos de valor maximo
    data_normal = Dataset_csv(path_data=[path_data_train_all[0], path_data_test_all[0]], random=False)
    Damax = data_normal.amax
    del data_normal

    # utils.generate_max_csvData([path_data_train_all[0], path_data_test_all[0]], path+'maximo.csv', has_label=True)
    # Damax = utils.load_max_csvData(path+'maximo.csv')

    c = tf.ConfigProto()
    c.gpu_options.visible_device_list = "1,2"
    # -------------------------------------------------------------------
    # ENTRENAMOS EL AUTOENCODER CON AMBAS CLASES - GENERAMOS UN PESO BASE
    # -------------------------------------------------------------------
    print('AE TRAIN ALL')
    print('------------')

    data_train = Dataset_csv(path_data=path_data_train_all, minibatch=mini_batch_train, max_value=Damax)
    print('Load data train...')
    data_test = Dataset_csv(path_data=path_data_test_all, minibatch=mini_batch_test, max_value=Damax, random=False)
    print('Load data test...')

    with tf.Session(config=c) as sess:

        x_batch = tf.placeholder(tf.float32, [None, dim_input])

        AEncode = AE.AEncoder(path_load_weight_all, learning_rate=learning_rate_all)
        AEncode.build(x_batch, layers)
        sess.run(tf.global_variables_initializer())

        print('Original Cost: ', test_model(AEncode, sess, data_test))
        train_model(AEncode, sess, data_train, objDatatest=data_test, epoch=epoch_all)

        # SAVE WEIGHTs
        AEncode.save_npy(sess, path_save_weight_all)

    del AEncode
    del data_train
    del data_test

    # -------------------------------------------------------------------
    #       ENTRENAMOS EL AUTOENCODER CON LA CLASE 0 - BENIGNO
    # -------------------------------------------------------------------
    print()
    print('AE TRAIN CLASS N')
    print('----------------')

    with tf.Session(config=c) as sess:

        x_batch = tf.placeholder(tf.float32, [None, dim_input])

        for i in range(num_class):
            data_train = Dataset_csv(path_data=path_data_train_class[i], minibatch=mini_batch_train, max_value=Damax, restrict=False)
            data_test = Dataset_csv(path_data=path_data_test_class[i], minibatch=mini_batch_test, max_value=Damax,restrict=False, random=False)

            AEncode = AE.AEncoder(path_load_weight[i], learning_rate=learning_rate_class)
            AEncode.build(x_batch, layers)
            sess.run(tf.global_variables_initializer())

            print('Original Cost CLASS '+str(i)+': ', test_model(AEncode, sess, data_test))
            train_model(AEncode, sess, data_train, objDatatest=data_test, epoch=epoch_class)

            # SAVE WEIGHTs
            AEncode.save_npy(sess, path_save_weight[i])

            print('\n------------------------------------------------------')
            del AEncode
            del data_train
            del data_test

    # -------------------------------------------------------------------
    #                   CLASIFICACION CON AUTOENCODERS
    # -------------------------------------------------------------------
    print()
    print('AE CLASSIFICATION')
    print('-----------------')

    data_test_dual = Dataset_csv(path_data=path_data_test_all, minibatch=1, max_value=Damax, restrict=False, random=False)

    with tf.Session(config=c) as sess:

        x_batch = tf.placeholder(tf.float32, [None, dim_input])
        AEclass = []

        for i in range(num_class):
            AEclass.append(AE.AEncoder(path_save_weight[i], learning_rate=learning_rate_class))
            AEclass[i].build(x_batch, layers)

        sess.run(tf.global_variables_initializer())
        test_model_all(AEclass, sess, data_test_dual, num_class)

    # -------------------------------------------------------------------
    #                          GENERAMOS CSV DE LA CAPA X
    # -------------------------------------------------------------------
    print()
    print('SAVE LAYER')
    print('----------')

    with tf.Session(config=c) as sess:

        x_batch = tf.placeholder(tf.float32, [None, dim_input])

        for i in range(num_class):
            data_train = Dataset_csv(path_data=path_data_train_class[i], minibatch=mini_batch_train,
                                     max_value=Damax, restrict=False)
            data_test = Dataset_csv(path_data=path_data_test_class[i], minibatch=mini_batch_test, max_value=Damax,
                                    restrict=False, random=False)

            AEncode = AE.AEncoder(path_save_weight[i], learning_rate=learning_rate_class)
            AEncode.build(x_batch, layers)
            sess.run(tf.global_variables_initializer())

            print('Original Cost CLASS ' + str(i) + ': ', test_model(AEncode, sess, data_test, index=i))
            print('\n------------------------------------------------------')
            del AEncode
            del data_train
            del data_test
