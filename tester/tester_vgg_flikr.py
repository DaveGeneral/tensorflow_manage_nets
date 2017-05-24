"""
Expert tester for the vgg19_trainable

:param load_weight_fc
# Variable para cargar los pesos de la capa fullConnect
# False : Cuando modificamos las capas fc para que el modelo clasifique un nuevo tipo de datos
# True  : Cuando ya contamos con un archivo de pesos entrenados .npy de la nueva red podemos cargarlos
# Nota, siempre que utilicemos inicialmente los pesos originales del archivo vgg19.npy debemos setear la variable en False
# ya que este archivo almacena los pesos de la red vgg original, al cargarlos en nuestra red ocurrira un error.

"""
import time
import tensorflow as tf
import sys, os
import numpy as np

switch_server = True

testdir = os.path.dirname('__file__')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

if switch_server is True:
    from tools import utils
    from nets import net_alex as vgg19
    from tools.dataset_image_mlabel import Dataset
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.nets import net_alex as vgg19
    from tensorflow_manage_nets.tools.dataset_image_mlabel import Dataset

# ..................................................................
# path = '../../data/ISB2016/'
# path_dir_image_train = path + "image_train_complete/"
# path_dir_image_test = path + "image_test_complete/"
# path_data_train = path + 'ISB_Train_complete.csv'
# path_data_test = path + 'ISB_Test_complete.csv'
#
# # VALIDATE INPUT DATA
# assert os.path.exists(path), 'No existe el directorio de datos ' + path
# assert os.path.exists(path_data_train), 'No existe el archivo con los datos de entrenamiento ' + path_data_train
# assert os.path.exists(path_data_test), 'No existe el archivo con los datos de pruebas ' + path_data_test

# GLOBAL VARIABLES
path = '../../data/flikr25k/flikr25k/'
path_dir_image_train = path + "data_all/"
path_dir_image_test = path + "data_all/"
path_data_train = path + 'flikr25_train_label.csv'
path_data_test = path + 'flikr25_test_label.csv'

# VALIDATE INPUT DATA
assert os.path.exists(path), 'No existe el directorio de datos ' + path
assert os.path.exists(path_data_train), 'No existe el archivo con los datos de entrenamiento ' + path_data_train
assert os.path.exists(path_data_test), 'No existe el archivo con los datos de pruebas ' + path_data_test


# Función, fase de test
def test_model(net, sess_test, objData):
    total = objData.total_images
    prob_predicted = np.random.random((0, net.num_class))

    print('\n# PHASE: Test classification')
    for i in range(objData.total_batchs_complete):
        batch, label = objData.generate_batch()
        prob, layer = sess_test.run([net.prob, net.fc7], feed_dict={vgg_batch: batch, train_mode: False})

        # save output of a layer
        # utils.save_layer_output(layer, label, name='layer_128', dir='../data/features/')
        # utils.save_layer_output_by_class(layer, label, name='layer_128', dir='../data/features/')

        # Acumulamos los aciertos de cada iteracion, para despues hacer un promedio
        predicted = utils.print_accuracy_multilabel(label, prob)
        prob_predicted = np.concatenate((prob_predicted, predicted), axis=0)

        objData.next_batch_test()

    # promediamos la presicion total
    print('\n# STATUS:')
    y_true = objData.labels.as_matrix()
    y_prob = prob_predicted

    hamming_score = utils.metrics_multiLabel(y_true, y_prob)
    return hamming_score


# Funcion, fase de entrenamiento
def train_model(net, sess_train, objData, epoch):
    print('\n# PHASE: Training model')
    for ep in range(epoch):
        print('\n     Epoch:', ep)
        t0 = time.time()
        cost_i = 0
        for i in range(objData.total_batchs):
            t_start = time.time()
            batch, label = objData.generate_batch()

            # Generate the 'one hot' or labels
            # label = tf.one_hot([li for li in label], on_value=1, off_value=0, depth=net.num_class)
            # label = list(sess_train.run(label))
            print(np.shape(batch), np.shape(label))
            # Run training
            _, cost = sess_train.run([net.train, net.cost],
                                     feed_dict={vgg_batch: batch, vgg_label: label, train_mode: True})
            # Next slice batch
            objData.next_batch()
            t_end = time.time()
            cost_i = cost_i + cost
            print("        > Minibatch: %d train on batch time: %7.3f seg." % (i, (t_end - t_start)))

        t1 = time.time()
        print("        Cost per epoch: ", cost_i)
        print("        Time epoch: %7.3f seg." % (t1 - t0))
        print("        Time per iteration: %7.3f seg." % ((t1 - t0) / epoch))


if __name__ == '__main__':
    # LOad y save  weights
    path_load_weight = '../weight/vgg19.npy'
    path_save_weight = '../weight/save_flikr_1.npy'
    load_weight_fc = False

    # Ultimas capas de la red
    last_layers = [4096, 24]

    epoch = 2
    mini_batch_train = 20
    mini_batch_test = 30
    learning_rate = 0.00001
    accuracy = 0

    # GENERATE DATA
    data_train = Dataset(path_data=path_data_train, path_dir_images=path_dir_image_train, minibatch=mini_batch_train,
                         cols=[0, [1, 24]], xtype='.jpg')
    data_test = Dataset(path_data=path_data_test, path_dir_images=path_dir_image_test, minibatch=mini_batch_test,
                        cols=[0, [1, 24]], random=False, xtype='.jpg')
    # data_test = Dataset(path_data=path_data_train, path_dir_images=path_dir_image_train, minibatch=mini_batch_train, cols=[0, [1,24]], multilabel=True, random=False, xtype='.jpg')

    with tf.Session() as sess:
        # DEFINE MODEL
        vgg_batch = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg_label = tf.placeholder(tf.float32, [None, last_layers[1]])
        train_mode = tf.placeholder(tf.bool)

        # Initialize of the model VGG19
        vgg = vgg19.VGG19(path_load_weight, learning_rate=learning_rate, load_weight_fc=load_weight_fc)
        vgg.build(vgg_batch, vgg_label, train_mode, last_layers=last_layers, softmax=False)
        sess.run(tf.global_variables_initializer())

        # # Execute Network
        test_model(net=vgg, sess_test=sess, objData=data_test)
        train_model(net=vgg, sess_train=sess, objData=data_train, epoch=epoch)
        accuracy = test_model(net=vgg, sess_test=sess, objData=data_test)

        # SAVE LOG: Genera un registro en el archivo log-server.txt
        utils.write_log(total_data=data_train.total_images,
                        epoch=epoch,
                        m_batch=mini_batch_train,
                        l_rate=learning_rate,
                        accuracy=accuracy,
                        file_npy=path_load_weight,
                        extra=str(last_layers))

        # SAVE WEIGHTs
        vgg.save_npy(sess, path_save_weight)




