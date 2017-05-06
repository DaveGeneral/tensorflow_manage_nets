import time
import tensorflow as tf
import sys, os
import numpy as np
import matplotlib.pyplot as plt

switch_server = True

testdir = os.path.dirname(__file__)
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

if switch_server is True:
    from tools import utils
    from nets import net_aencoder as AE
else:
    from tensorflow_manage_nets2.tools import utils
    from tensorflow_manage_nets2.nets import net_aencoder as AE
# ..................................................................


# Plot example reconstructions
def plot_result(net, sess, batch, n_examples=5):
    print('Plotting')
    mask_np = np.random.binomial(1, 1 - net.noise, batch.shape)
    recon = sess.run(net.y, feed_dict={x_batch: batch, mask: mask_np, noise_mode: False})
    fig, axs = plt.subplots(2, n_examples, figsize=(n_examples, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(np.reshape(batch[example_i, :], (28, 28)))
        axs[1][example_i].imshow(np.reshape(recon[example_i, :], (28, 28)))

    fig.show()
    plt.draw()
    plt.waitforbuttonpress()
# ..................................................................


if __name__ == '__main__':

    path_load_weight = '../weight/saveAE_1.npy'
    path_save_weight = '../weight/saveAE_1.npy'

    mini_batch_train = 20
    mini_batch_test = 25
    epoch = 5
    learning_rate = 0.0001
    noise_level = 0

    # Datos
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    with tf.Session() as sess:

        x_batch = tf.placeholder(tf.float32, [None, 784])
        mask = tf.placeholder(tf.float32, [None, 784])
        noise_mode = tf.placeholder(tf.bool)

        AEncode = AE.AEncoder(path_load_weight, learning_rate=learning_rate, noise=noise_level)
        AEncode.build(x_batch, mask, noise_mode, [500])
        sess.run(tf.global_variables_initializer())

        # Test inicial
        test_xs = mnist.test.images
        mask_np = np.random.binomial(1, 1 - noise_level, test_xs.shape)
        print('Original: ', sess.run(AEncode.cost, feed_dict={x_batch: test_xs, mask: mask_np, noise_mode: False}))
        total = len(trX)

        # Train red
        for i in range(epoch):
            for start, end in zip(range(0, total, 128), range(128, total, 128)):
                input_ = trX[start:end]
                mask_np = np.random.binomial(1, 1 - noise_level, input_.shape)
                sess.run(AEncode.train, feed_dict={x_batch: input_, mask: mask_np, noise_mode: False})

            mask_np = np.random.binomial(1, 1 - noise_level, teX.shape)
            print(i, sess.run(AEncode.cost, feed_dict={x_batch: teX, mask: mask_np, noise_mode: False}))

        # Save
        AEncode.save_npy(sess, path_save_weight)

        # Plot example reconstructions
        n_examples = 10
        plot_result(AEncode, sess, mnist.test.next_batch(n_examples)[0], n_examples)
