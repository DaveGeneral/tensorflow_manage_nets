import sys, os
import numpy as np
from numpy import genfromtxt
from sklearn import neighbors, datasets, model_selection
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA

switch_server = True
testdir = os.path.dirname('__file__')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

# from onehotCNN.tune_subspace import getfractal
# from onehotCNN.tune_subspace import computelshparams
# Add other method here!
from onehotCNN.autoe import AUTOE
from onehotCNN.svd import SVD
from onehotCNN.cp import CP
from onehotCNN.dct import DCT
from onehotCNN.dwt import DWT
from onehotCNN.ipla import IPLA
from onehotCNN.paa import PAA
from onehotCNN.sax import SAX

if switch_server is True:
    from tools import utils
    from nets import net_aencoder as AE
    from tools.dataset_csv import Dataset_csv
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.nets import net_aencoder as AE
    from tensorflow_manage_nets.tools.dataset_csv import Dataset_csv


xpath = '../data/reduceDimension/'


def path_datasets(opc):

    if opc == 0:
        # MNIST
        data_name = 'MNIST'
        total = 10000
        path_data_test = [xpath + data_name + '/' + 'mnist-test-800.csv']
        path_data_train = [xpath + data_name + '/' + 'mnist-train-800.csv']
        path_max = xpath + data_name + '/' +'max-mnist.csv'
        dims = [63,94,141]

    elif opc == 1:
        # CIFAR
        data_name = 'CIFAR10'
        total = 10000
        path_data_test = [xpath + data_name + '/' + 'cifar10-test-4096.csv']
        path_data_train = [xpath + data_name + '/' + 'cifar10-train-4096.csv']
        path_max = xpath + data_name + '/' + 'max-cifar10.csv'
        dims = [94, 141, 211]

    elif opc == 2:
        # SVHN
        data_name = 'SVHN'
        total = 26032
        path_data_test = [xpath + data_name + '/' + 'svhn-test-1152.csv']
        path_data_train = [xpath + data_name + '/' + 'svhn-train-1152.csv']
        path_max = xpath + data_name + '/' + 'max-svhn.csv'
        dims = [42,63,94]

    elif opc == 3:
        # AGNews
        data_name = 'AGNEWS'
        total = 7552
        path_data_test = [xpath + data_name + '/' + 'agnews-test-8704.csv']
        path_data_train = [xpath + data_name + '/' + 'agnews-train-8704.csv']
        path_max = xpath + data_name + '/' + 'max-agnews.csv'
        dims = [94, 141, 211]

    return path_data_train, path_data_test, path_max, data_name, dims


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


def get_data_all(path_data, array_max):

    data_all = Dataset_csv(path_data=path_data, max_value=array_max)
    data_all.set_minibatch(data_all.total_inputs)
    X_data, y_data = data_all.generate_batch()

    return X_data, y_data, len(y_data)


def make_reduce_matrix(path_data, xmethod, dim_optimal, dataname, extraname):

    matrix = genfromtxt(path_data, delimiter=',')
    shape = np.shape(matrix)

    X_data = matrix[:, :shape[1] - 1]
    y_data = matrix[:, -1:]
    total_data = len(y_data)

    reducedMatrix = reduce_dimension_function(xmethod, X_data, dim_optimal)
    filename_rm = xpath + dataname + '/' + dataname.lower() + '-' + extraname + '-' + xmethod + '-' + str(dim_optimal) + '.csv'

    print('filename_pca:', filename_rm)
    f = open(filename_rm, "w")
    for i in range(total_data):
        f.write(",".join(map(str, np.concatenate((reducedMatrix[i], y_data[i]), axis=0))) + "\n")
    f.close()
    print('Save!!')

    return filename_rm


if __name__ == '__main__':

    for opc in range(4):
        path_data_train_csv, path_data_test_csv, path_max_csv, name, dims = path_datasets(opc)

        for xdim in dims:
            a = make_reduce_matrix(path_data_test_csv[0], 'pca', xdim, name, 'test')
            b = make_reduce_matrix(path_data_train_csv[0], 'pca', xdim, name, 'train')
            utils.normalization_complete([a, b])



