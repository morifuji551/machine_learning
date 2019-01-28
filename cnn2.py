import os
import tensorflow as tf 
from keras.datasets import mnist 
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#######  DEFINE CNN PARAMETERS  ##########
height = 28
width = 28
channels = 1
classes = 10
n_inputs = height * width 

n_epochs = 10
batch_size = 100

conv1_fmaps = 50 
conv1_ksize = 5 
conv1_stride = 1 
conv1_pad = "SAME"
conv1_activation = tf.nn.relu

pool1_ksize = [1,2,2,1]
pool1_strides = [1,2,2,1]
pool1_padding = "VALID"

conv2_fmaps = 50 
conv2_ksize = 5
conv2_stride = 1
conv2_pad = "SAME" 
conv2_activation = tf.nn.relu 

pool2_ksize = [1,2,2,1]
pool2_strides = [1,2,2,1]
pool2_pad = "VALID"

flatten_size = [-1, 7*7*conv2_fmaps]

Dense1_neurons = 500
Dense1_activation = tf.nn.relu 

Dense2_neurons = classes 
#########################################

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name = "X")
    X_reshaped = tf.reshape(X, [-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape = [None], name = "y")

with tf.name_scope("conv1"):
    conv1 = tf.layers.conv2d(X_reshaped, 
                            filters = conv1_fmaps, 
                            kernel_size = conv1_ksize, 
                            padding = conv1_pad,
                            activation = conv1_activation
                            )

with tf.name_scope("pool1"):
    pool1 = tf.nn.max_pool(conv1,
                            ksize = pool1_ksize,
                            strides = pool1_strides,
                            padding = pool1_padding
                            )

with tf.name_scope("conv2"):
    conv2 = tf.layers.conv2d(pool1,
                            filters = conv2_fmaps,
                            kernel_size = conv2_ksize,
                            padding = conv2_pad,
                            activation = conv2_activation
                            )

with tf.name_scope("pool2"):
    pool2 = tf.nn.max_pool(conv2,
                            ksize = pool2_ksize,
                            strides = pool2_strides,
                            padding = pool2_pad
                            )

with tf.name_scope("flatten"):
    pool2_flat = tf.reshape(pool2, shape = flatten_size)

with tf.name_scope("dense1"):
    dense1 = tf.layers.dense(pool2_flat,
                            Dense1_neurons,
                            activation = Dense1_activation)

with tf.name_scope("output"):
    logits = tf.layers.dense(dense1, Dense2_neurons,)
    y_proba = tf.nn.softmax(logits)

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, k = 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init"):
    init = tf.global_variables_initializer()

def preprocess(data, label = False):
    if label:
        data = data.astype(np.int32)
    else:
        data = data.astype(np.float32).reshape(-1,28*28) / 255.0
    return data 

#fetch data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = [preprocess(d) for d in [X_train, X_test]]
y_train, y_test = [preprocess(d, label = True) for d in [y_train, y_test]]
X_val, X_train = X_train[:5000], X_train[5000:]
y_val, y_train = y_train[:5000], y_train[5000:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
