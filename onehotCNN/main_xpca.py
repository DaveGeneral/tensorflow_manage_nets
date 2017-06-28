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

# Add other method here!
from autoe import AUTOE
from svd import SVD
from cp import CP
from dct import DCT
from dwt import DWT
from ipla import IPLA
from paa import PAA
from sax import SAX

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
path_data_status = 'saveStatus/'
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


def reduce_dimension_function(option, X_train, new_dim):

    if option == 'pca':
        n_batches = 10
        pca = PCA(n_components=new_dim)
        pca.fit(X_train)
        X_reduced = pca.transform(X_train)
        print(np.shape(X_reduced))
        return X_reduced

    elif option == 'autoencoder':
        autoe = AUTOE()
        autoe.set_data(X_train)
        autoe.shuffle_data()
        # autoe.normalize(-1.0, 1.0)
        autoe.divide_data(0.8)
        autoe.create_autoencoder(new_dim)
        # autoe.normalize() # best results of clustering for interval [0, 1]
        # autoe.standardize()
        autoe.train_autoencoder()
        # autoe.test_autoencoder()
        # autoe.get_activations()
        autoe.sort_activations()

        # autoe.plot_reconstruction(i+1)
        # autoe.save_activations('caract_autoe.csv')
        # autoe.save_activations(filename+'_'+str(i+1)+'.csv')
        # autoe.save_activations('caract_autoe.csv')
        return autoe.get_activations()

    elif option == 'svd':
        svd = SVD()
        svd.set_data(X_train)
        # svd.load_data('dataset.csv')
        svd.shuffle_data()
        # svd.normalize(-1.0,1.0)
        # svd.standardize()
        svd.run_svd(new_dim)
        svd.sort_coefficients()
        # svd.save_activations('caract_'+svd.__class__.__name__.lower()+'60.csv')
        # svd.save_activations(filename+'_'+str(i+1)+'.csv')
        return svd.get_coefficients()

    elif option == 'cp':
        cp = CP()
        cp.set_data(X_train)
        # cp.load_data('dataset.csv')
        cp.shuffle_data()
        # cp.normalize(-1.0, 1.0)
        # cp.standardize()
        cp.execute_cp(new_dim)
        cp.sort_coefficients()
        # cp.save_activations(filename+'_'+str(i+1)+'.csv')
        # cp.save_activations('caract_cp.csv')
        return cp.get_coefficients()

    elif option == 'dct':
        dcost = DCT()
        dcost.set_data(X_train)
        dcost.shuffle_data()
        # dcost.normalize(-1.0, 1.0)
        dcost.execute_dct(new_dim)
        dcost.sort_coefficients()
        # dcost.save_activations(filename+'_'+str(i+1)+'.csv')
        # dcost.save_activations('caract_dct.csv')
        return dcost.get_coefficients()

    elif option == 'dwt':
        dwt = DWT()
        dwt.set_data(X_train)
        dwt.shuffle_data()
        # dwt.normalize(-1,1)
        # dwt.standardize()
        dwt.execute_dwt(new_dim)
        dwt.sort_coefficients()
        return dwt.get_coefficients()

    elif option == 'ipla':
        paa = IPLA()
        paa.set_data(X_train)
        # paa.load_data('dataset.csv')
        paa.shuffle_data()
        # paa.normalize()
        # paa.standardize()
        paa.execute_ipla(new_dim)
        paa.sort_coefficients()
        return paa.get_coefficients()

    elif option == 'paa':
        paa = PAA()
        paa.set_data(X_train)
        # paa.load_data('dataset.csv')
        paa.shuffle_data()
        # paa.normalize(-1, 1)
        # paa.standardize()
        paa.execute_paa(new_dim)
        paa.sort_coefficients()
        return paa.get_coefficients()

    elif option == 'sax':
        sax = SAX()
        sax.set_data(X_train)
        sax.shuffle_data()
        # sax.normalize()
        # sax.standardize()
        sax.execute_sax(new_dim)
        sax.sort_coefficients()

        return sax.get_coefficients()

    else:
        return 'Invalid option'


def normalize(dataset, mini, maxi):
    data = dataset.T
    # print 'data.shape =zzz ', data.shape
    data_norm = (data * 1.0 - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data_norm = scaler.fit_transform(data)
    data_norm = data_norm * (maxi - mini) + mini
    dataset = data_norm.T
    return dataset


if __name__ == '__main__':

    ratio_diff = 0.05
    # funcOpc = {'pca':[-1.0,0.0,[],[]], 'autoencoder':[-1.0,0.0,[],[]], 'svd':[-1.0,0.0,[],[]],
    #            'cp':[-1.0,0.0,[],[]], 'dct':[-1.0,0.0,[],[]], 'dwt':[-1.0,0.0,[],[]],
    #            'ipla':[-1.0,0.0,[],[]], 'paa':[-1.0,0.0,[],[]], 'sax':[-1.0,0.0,[],[]]}

    funcOpc = {'pca': [-1.0, 0.0, [], []], 'svd': [-1.0, 0.0, [], []], 'dct': [-1.0, 0.0, [], []],
               'dwt': [-1.0, 0.0, [], []], 'ipla': [-1.0, 0.0, [], []], 'paa': [-1.0, 0.0, [], []]}

    for x in range(4, 5):
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
        cen_ratio_diff = True
        l_hidden = 4

        csv_setname = path_data_test[0].split('/')[-1]
        print('csv_setname: ', csv_setname)
        reducedMatrix = None

        results_fn = path_data_status + setname + '.function_fractal'
        f = open(results_fn, 'a')
        output = [setname, dim, XFractal]
        f.write(','.join(map(str, output)) + '\n')

        while l_hidden < int(dim_input/2) and cen_ratio_diff is True:

            cen_ratio_diff = False
            print('\n[PRUEBA :', l_hidden, ']')

            for opcF in funcOpc:
                t0 = time.time()
                print('     --', opcF, '--')
                print('     -Dim reduce...')
                reducedMatrix = reduce_dimension_function(opcF, Xmatrix, l_hidden)
                print('     -Get fractal...')
                dimFractal = getfractal(path, opcF + '_' + csv_setname, reducedMatrix)
                funcOpc[opcF][0] = funcOpc[opcF][1]
                funcOpc[opcF][1] = dimFractal
                #
                funcOpc[opcF][2].append(l_hidden)
                funcOpc[opcF][3].append(dimFractal)

                if abs(funcOpc[opcF][1] - funcOpc[opcF][0]) > ratio_diff:
                    cen_ratio_diff = True

                t1 = time.time() - t0
                print("     func", opcF + ':', l_hidden, 'dim_old:', funcOpc[opcF][0], 'dim_new:', funcOpc[opcF][1],
                      'time:', t1)
                output = [setname, opcF, l_hidden, funcOpc[opcF][0], funcOpc[opcF][1], t1]
                f.write(','.join(map(str, output)) + '\n')

            total_time = (time.time() - t0)
            print("total_time:", total_time)
            l_hidden = l_hidden + int(l_hidden / 2)  # next step

        for opcF in funcOpc:
            f.write(','.join(map(str, [setname, opcF])) + '\n')
            f.write(','.join(map(str, funcOpc[opcF][2])) + '\n')
            f.write(','.join(map(str, funcOpc[opcF][3])) + '\n')

        f.close()
        print('condition: ', (l_hidden < int(dim_input/2)), cen_ratio_diff)
        print("-------------------------------")
        # print(input())

    print('Finish Dataset!!!')
    print("-------------------------------")
    print("-------------------------------")



