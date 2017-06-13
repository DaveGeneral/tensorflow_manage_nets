import time
import tensorflow as tf
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tune_subspace import getfractal
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt
import numpy as np

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
def test_model(net, sess_test, objData, lh=0, index=0):
    total = objData.total_inputs
    cost_total = 0

    matrix = np.random.random((0, lh))

    for i in range(objData.total_batchs_complete):
        x_, label = objData.generate_batch()
        cost, layer = sess_test.run([net.cost, net.z], feed_dict={x_batch: x_})

        matrix = np.concatenate((matrix, layer), axis=0)

        #         utils.save_layer_output(layer, label, name='endoce_relu', dir='../data/MNIST_data/test64/')
        #         utils.save_layer_output(layer, label, name='endoce_relu_class'+str(index), dir='../data/MNIST_data/test64/')
        # print(np.shape(layer))
        cost_total = cost_total + cost
        objData.next_batch_test()

    return cost_total, cost_total / total, matrix


# Funcion, fase de entrenamiento
def train_model(net, sess_train, objData, objDatatest, epoch):
    print('\n# PHASE: Training model')
    for ep in range(epoch):
        for i in range(objData.total_batchs):
            batch, _ = objData.generate_batch()
            sess_train.run(net.train, feed_dict={x_batch: batch})
            objData.next_batch()

            # cost_tot, cost_prom, _ = test_model(net, sess_train, objDatatest)
            # print('     Epoch', ep, ': ', cost_tot, ' / ', cost_prom)


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

    # print(np.shape(y_true))
    #     ax = np.reshape(objData.labels.values, [objData.total_inputs])
    #     ax = list(ax)
    #     f = open("PaperCIARP.csv", "a+")
    #     f.write(",".join(map(str, ax)) + "\n")
    #     f.write(",".join(map(str, y_result)) + "\n")
    #     f.close()

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

# GLOBAL VARIABLES
opc = 2

if opc == 0:

    # DATA MNIST
    dim_input = 800
    path = '../data/MNIST_dataA/'
    path_data_train = [path + 'train800/' + 'output_train-mnist-800.csv']
    path_data_test = [path + 'test800/' + 'output_test-mnist-800.csv']
    path_maximo = path + 'maximo_mnist800.csv'

    path_weight = '../weight/onehotCNN/'
    path_load_weight = None
    path_save_weight = path_weight + 'save_ae_mnist.npy'

elif opc == 1:

    # DATA agNews
    dim_input = 8704
    path = '../data/agnews/'
    path_data_train = [path + 'output_train_news_8704.csv']
    path_data_test = [path + 'output_test_news_8704.csv']
    path_maximo = path + 'maximo_agnews.csv'

    path_weight = '../weight/onehotCNN/'
    path_load_weight = None
    path_save_weight = path_weight + 'save_ae_agnews.npy'

elif opc == 2:

    # Data CIFAR
    dim_input = 4096
    path = '../data/features_cifar10_vgg/'
    path_data_train = [path + 'output_trainVGG_relu6.csv']
    path_data_test = [path + 'output_testVGG_relu6.csv']
    path_maximo = path + 'maximo.csv'

    path_weight = '../weight/onehotCNN/'
    path_load_weight = None
    path_save_weight = path_weight + 'save_ae_cifar10.npy'

elif opc == 3:

    # DATA SVHN
    dim_input = 1152
    path = '../data/SVHN_data/'
    path_data_train = [path + 'train1152/' + 'output_train_SVHN.csv']
    path_data_test = [path + 'test1152/' + 'output_test_SVHN.csv']
    path_maximo = path + 'maximo_svhn1152.csv'

    path_weight = '../weight/onehotCNN/'
    path_load_weight = None
    path_save_weight = path_weight + 'save_ae_SVHN.npy'

elif opc == 4:

    # DATA ISBI dim_input = 4096, layers = [[2048,'sigmoid'], [512,'sigmoid']]
    dim_input = 4096
    path = '../data/features/testskin1/muestraA/'
    path_data_train = [path + 'SKINfeaturesA_Train.csv']
    path_data_test = [path + 'SKINfeaturesA_Test.csv']
    path_maximo = path + 'maximo_ISBI.csv'

    path_weight = '../weight/onehotCNN/'
    path_load_weight = None
    path_save_weight = path_weight + 'save_ae_ISBI.npy'

assert os.path.exists(path), print('No existe el directorio de datos ' + path)
assert os.path.exists(path_weight), print('No existe el directorio de pesos ' + path_weight)

from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA


def reduce_using_incpca(X_train, new_dim):
    n_batches = 10
    inc_pca = IncrementalPCA(n_components=new_dim)
    for X_batch in np.array_split(X_train, n_batches):
        inc_pca.partial_fit(X_batch)
    X_reduced = inc_pca.transform(X_train)
    return X_reduced


def reduce_using_pca(X_train, new_dim):
    n_batches = 10
    pca = PCA(n_components=new_dim)
    pca.fit(X_train)
    X_reduced = pca.transform(X_train)
    print(np.shape(X_reduced))
    return X_reduced


if __name__ == '__main__':

    mini_batch_train = 35
    mini_batch_test = 5
    epoch = 50
    learning_rate = 0.00005
    l_hidden = 16
    ratio_diff = 0.05
    pca = True

    Damax = utils.load_max_csvData(path_maximo)
    # data_train = Dataset_csv(path_data=path_data_train, minibatch=mini_batch_train, max_value=Damax, restrict=False)
    data_test = Dataset_csv(path_data=path_data_test, minibatch=mini_batch_test, max_value=Damax, restrict=False,
                            random=False)

    Xmatrix = genfromtxt(path_data_train[0], delimiter=',')
    shape = np.shape(Xmatrix)
    Xmatrix = Xmatrix[:, :shape[1] - 1]

    XFractal = getfractal(path, path_data_test[0].split('/')[-1], Xmatrix)
    print("fractal dimension of X:", XFractal)

    c = tf.ConfigProto()
    c.gpu_options.visible_device_list = "0"
    with tf.Session() as sess:

        x_batch = tf.placeholder(tf.float32, [None, dim_input])

        oldF = 0.0
        newF = 0.0
        cen = True
        i = 0
        # and (newF < XFractal*0.9 )
        while l_hidden < dim_input and (cen is True or abs(newF - oldF) > ratio_diff) and (newF < XFractal * 0.9):
            print('\n[PRUEBA :', l_hidden, ']')
            cen = False
            t0 = time.time()


            def reduce_using_autoencoders(new_dim):
                layers = [[new_dim, 'relu']]
                AEncode = AE.AEncoder(path_load_weight, learning_rate=learning_rate)
                AEncode.build(x_batch, layers)
                sess.run(tf.global_variables_initializer())

                train_model(AEncode, sess, data_test, objDatatest=data_test, epoch=epoch)
                _, _, matrix = test_model(AEncode, sess, data_test, new_dim)
                print(np.shape(matrix))
                return matrix


            if pca == True:
                reducedMatrix = reduce_using_pca(Xmatrix, l_hidden)
            else:
                matrix = reduce_using_autoencoders(l_hidden)
                # SAVE WEIGHTs
                AEncode.save_npy(sess, path_save_weight)
                del AEncode
                reducedMatrix = matrix

            dimFractal = getfractal(path, path_data_test[0].split('/')[-1], reducedMatrix)

            oldF = newF
            newF = dimFractal

            i = i + 1
            print("iter:", l_hidden, oldF, newF)
            l_hidden = l_hidden * 2
            total_time = (time.time() - t0)
            print("total_time:", total_time)
            print("-------------------------------")

    print('Finish!!!')






