import time
import tensorflow as tf
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tune_subspace import getfractal
from tune_subspace import computelshparams
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

# GLOBAL VARIABLES
# opc = 2

path_data = '../data/onehotCNN/'
path_weight = '../weight/onehotCNN/'
path_load_weight = None
path_save_weight = path_weight + 'save_ae_mnist.npy'


def path_datasets(opc):
    if opc == 0:

        # DATA MNIST
        dim_input = 800
        path = '../data/MNIST_dataA/'
        path_data_train = [path + 'train800/' + 'output_train-mnist-800.csv']
        path_data_test = [path + 'test800/' + 'output_test-mnist-800.csv']
        path_maximo = path + 'maximo_mnist800.csv'
        datasetname = 'mnist'

    elif opc == 1:

        # Data CIFAR
        dim_input = 4096
        path = '../data/features_cifar10_vgg/'
        path_data_train = [path + 'output_trainVGG_relu6.csv']
        path_data_test = [path + 'output_testVGG_relu6.csv']
        path_maximo = path + 'maximo.csv'
        datasetname = 'cifar10'


    elif opc == 2:

        # DATA SVHN
        dim_input = 1152
        path = '../data/SVHN_data/'
        path_data_train = [path + 'train1152/' + 'output_train_SVHN.csv']
        path_data_test = [path + 'test1152/' + 'output_test_SVHN.csv']
        path_maximo = path + 'maximo_svhn1152.csv'
        datasetname = 'svhn'

    elif opc == 3:

        # DATA ISBI
        dim_input = 4096
        path = '../data/features/testskin1/muestraA/'
        path_data_train = [path + 'SKINfeaturesA_Train.csv']
        path_data_test = [path + 'SKINfeaturesA_Test.csv']
        path_maximo = path + 'maximo_ISBI.csv'
        datasetname = 'isbi'

    elif opc == 4:

        # DATA agNews
        dim_input = 8704
        path = '../data/agnews/'
        path_data_train = [path + 'output_train_news_8704.csv']
        path_data_test = [path + 'output_test_news_8704.csv']
        path_maximo = path + 'maximo_agnews.csv'
        datasetname = 'agnews'

    elif opc == 5:

        # DATA serie temporal sintetica
        dim_input = 60
        path = '../data/syntheticChart/'
        # path_data_train = [path + 'output_train_news_8704.csv']
        path_data_test = [path + 'sChart.csv']
        path_maximo = path + 'maximo_serieTemporal.csv'
        datasetname = 'sChart'

    return dim_input, path, path_data_test, path_maximo, datasetname


# assert os.path.exists(path), print('No existe el directorio de datos ' + path)
# assert os.path.exists(path_weight), print('No existe el directorio de pesos ' + path_weight)

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

    learning_rate = 0.00005
    l_hidden = 16
    ratio_diff = 0.05

    for x in range(0, 6):
        opc = x
        print("DATASET", opc, ':')
        print("----------")
        dim_input, path, path_data_test, path_maximo, setname = path_datasets(opc)

        # Damax = utils.load_max_csvData(path_maximo)
        Xmatrix = genfromtxt(path_data_test[0], delimiter=',')
        shape = np.shape(Xmatrix)

        labels = Xmatrix[:, -1:]
        Xmatrix = Xmatrix[:, :shape[1] - 1]

        XFractal = getfractal(path, path_data_test[0].split('/')[-1], Xmatrix)
        print("fractal dimension of X:", XFractal)
        shape_M = np.shape(Xmatrix)
        dim = shape_M[1]
        oldF = -1.0
        newF = 0.0
        cen = True
        # l_hidden = 16
        l_hidden = 4
        csv_setname = path_data_test[0].split('/')[-1]
        print('csv_setname: ', csv_setname)
        reducedMatrix = None
        while l_hidden < dim_input and (abs(newF - oldF) > ratio_diff) and cen == True:
            print('\n[PRUEBA :', l_hidden, ']')
            t0 = time.time()

            oldReducedMatrix = reducedMatrix
            reducedMatrix = reduce_using_pca(Xmatrix, l_hidden)

            dimFractal = getfractal(path, csv_setname, reducedMatrix)

            oldF = newF
            newF = dimFractal

            print("iter:", l_hidden, oldF, newF)
            total_time = (time.time() - t0)
            print("total_time:", total_time)

            results_fn = path_data + setname + '.pca_fractal'
            output = [setname, l_hidden, newF, total_time]
            f = open(results_fn, 'a')
            f.write('\t'.join(map(str, output)) + '\n')
            f.close()

            print('condition: ', (l_hidden < dim_input), (abs(newF - oldF) > ratio_diff), (newF < XFractal * 0.9))

            print("-------------------------------")
            if newF < XFractal and (newF >= XFractal * 0.9):
                cen = False

            if (abs(newF - oldF) <= ratio_diff):
                l_hidden = (int)(l_hidden / 2)
                reducedMatrix = oldReducedMatrix

            l_hidden = l_hidden + int(l_hidden / 2)  # next step

            print(input())

        # write the fractal dimension of the original dataset
        results_fn = path_data + setname + '.pca_fractal'
        output = [setname, dim, XFractal, 0.0]
        f = open(results_fn, 'a')
        f.write('\t'.join(map(str, output)) + '\n')
        f.close()

        # save matrix
        print('Saving...')
        total = len(labels)
        filename_pca = path_data + setname + '-' + str(int(l_hidden / 2)) + "-pca.csv"
        print('filename_pca:', filename_pca)
        f = open(filename_pca, "w")
        for i in range(total):
            f.write(",".join(map(str, np.concatenate((reducedMatrix[i], labels[i]), axis=0))) + "\n")
        f.close()
        print('Save!!')

        # compute lshparams
        computelshparams(path_data, setname + '-' + str(int(l_hidden / 2)) + "-pca.csv", reducedMatrix)

        print('Finish Dataset!!!')
        print("-------------------------------")
        print("-------------------------------")

