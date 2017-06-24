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
        data_name = 'mnist'
        total = 10000
        path_data = [xpath + 'output_test-mnist-800.csv']
        path_max = xpath + 'maximo_mnist800.csv'

    elif opc == 1:
        # CIFAR
        data_name = 'cifar'
        total = 10000
        path_data = [xpath + 'output_testVGG_relu6.csv']
        path_max = xpath + 'maximo.csv'

    elif opc == 2:
        # SVHN
        data_name = 'svhn'
        total = 26032
        path_data = [xpath + 'output_test_SVHN.csv']
        path_max = xpath + 'maximo_svhn1152.csv'

    elif opc == 3:
        # AGNews
        data_name = 'agNews'
        total = 7552
        path_data = [xpath + 'output_test_news_8704.csv']
        path_max = xpath + 'maximo_agnews.csv'

    elif opc == 4:
        # ISBI
        data_name = 'isbi'
        total = 379
        path_data = [xpath + 'SKINfeaturesA_Test.csv']
        path_max = xpath + 'maximo_ISBI.csv'

    return path_data, path_max, total, data_name


if __name__ == '__main__':

    path_logs = xpath + 'resultClasify.csv'
    f = open(path_logs, 'a')

    for i in range(5):
        path_data_csv, path_max_csv, total_samples, name = path_datasets(i)

        print('\n[NAME:', name, ']')
        Damax = utils.load_max_csvData(path_max_csv)
        data_train = Dataset_csv(path_data=path_data_csv, minibatch=total_samples, max_value=Damax)
        data, label = data_train.generate_batch()

        X_train, X_test, y_train, y_test = model_selection.train_test_split(data, label, test_size=0.30, random_state=42)
        print(np.shape(X_train), np.shape(X_test))

        knn = neighbors.KNeighborsClassifier()
        print("     Train model...")
        knn.fit(X_train, y_train)
        print("     Test model...")
        Z = knn.predict(X_test)

        acc = utils.metrics_multiclass(y_test, Z)

        print('     Save result...')
        output = [name, total_samples, acc, path_data_csv]
        f.write(','.join(map(str, output)) + '\n')
        f.write(','.join(map(str, y_test)) + '\n')
        f.write(','.join(map(str, Z)) + '\n')

    f.close()
