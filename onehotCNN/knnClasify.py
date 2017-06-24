import sys, os
import numpy as np
import pylab as pl
from sklearn import neighbors, datasets, model_selection

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


xpath = '../data/onehotcnn/'


def path_datasets(opc):

    if opc == 0:
        # MNIST
        data_name = 'MNIST'
        total = 10000
        path_data_test = [xpath + 'output_test-mnist-800.csv']
        path_data_train = [xpath + 'output_train-mnist-800.csv']
        path_max = xpath + 'maximo_mnist800.csv'

    elif opc == 1:
        # CIFAR
        data_name = 'CIFAR-10'
        total = 10000
        path_data_test = [xpath + 'output_testVGG_relu6.csv']
        path_data_train = [xpath + 'output_trainVGG_relu6.csv']
        path_max = xpath + 'maximo.csv'

    elif opc == 2:
        # SVHN
        data_name = 'SVHN'
        total = 26032
        path_data_test = [xpath + 'output_test_SVHN.csv']
        path_data_train = [xpath + 'output_train_SVHN.csv']
        path_max = xpath + 'maximo_svhn1152.csv'

    elif opc == 3:
        # AGNews
        data_name = 'AG_NEWS'
        total = 7552
        path_data_test = [xpath + 'output_test_news_8704.csv']
        path_data_train = [xpath + 'output_train_news_8704.csv']
        path_max = xpath + 'maximo_agnews.csv'

    elif opc == 4:
        # ISBI
        data_name = 'ISBI'
        total = 379
        path_data_test = [xpath + 'SKINfeaturesA_Test.csv']
        path_data_train = [xpath + 'SKINfeaturesA_Train.csv']
        path_max = xpath + 'maximo_ISBI.csv'

    return path_data_train, path_data_test, path_max, data_name


def get_data_split(path_data, array_max, test_size=0.3):
    data_all = Dataset_csv(path_data=path_data, max_value=array_max)
    data_all.set_minibatch(data_all.total_inputs)
    data, label = data_all.generate_batch()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, label, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test, len(y_train), len(y_test)


def get_data_all(path_train, path_test, array_max):

    data_all = Dataset_csv(path_data=path_train, max_value=array_max)
    data_all.set_minibatch(data_all.total_inputs)
    X_train, y_train = data_all.generate_batch()

    data_all = Dataset_csv(path_data=path_test, max_value=array_max)
    data_all.set_minibatch(data_all.total_inputs)
    X_test, y_test = data_all.generate_batch()

    return X_train, X_test, y_train, y_test, len(y_train), len(y_test)


if __name__ == '__main__':

    path_logs = xpath + 'resultClasify2.csv'
    f = open(path_logs, 'a')

    for i in range(0,3):
        path_data_train_csv, path_data_test_csv, path_max_csv, name = path_datasets(i)

        print('\n[NAME:', name, ']')
        Damax = utils.load_max_csvData(path_max_csv)

        # Metodo 1
        # X_train, X_test, y_train, y_test, total_train, total_test = get_data_split(path_data_test_csv, Damax, 0.3)
        # Metodo 2
        X_train, X_test, y_train, y_test, total_train, total_test = get_data_all(path_data_train_csv, path_data_test_csv, Damax)
        print(np.shape(X_train), np.shape(X_test))

        knn = neighbors.KNeighborsClassifier()
        print("     Train model...")
        knn.fit(X_train, y_train)
        print("     Test model...")
        Z = knn.predict(X_test)

        acc = utils.metrics_multiclass(y_test, Z)

        print('     Save result...')
        output = [name, total_test, acc, path_data_test_csv]
        f.write(','.join(map(str, output)) + '\n')
        f.write(','.join(map(str, y_test)) + '\n')
        f.write(','.join(map(str, Z)) + '\n')

    f.close()
