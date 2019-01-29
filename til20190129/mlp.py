import tensorflow as tf 
import numpy as np 
from keras.datasets import mnist 
import keras

#### ml parameters #####

input_dsize = 28*28

batch_size = 128
epochs = 100

num_neurons1 = 300
num_neurons2 = 200
num_neurons3 = 200
classes = 10

########################

class hiddenlayer():
    def __init__(self, inputs, num_neurons, activation = False):
        self.inputs = inputs
        input_size = int(inputs.get_shape()[1])
        init_W = tf.truncated_normal([input_size, num_neurons], stddev = 0.05, dtype = tf.float32) 
        self.W = tf.Variable(initial_value = init_W, dtype = tf.float32)
        self.b = tf.Variable(tf.zeros([num_neurons]), dtype = tf.float32)
        self.activation = activation
    
    def output(self):
        if not(self.activation):
            return tf.matmul(self.inputs, self.W) + self.b
        else:
            return self.activation(tf.matmul(self.inputs, self.W) + self.b)

X = tf.placeholder(dtype = tf.float32, shape = [None, input_dsize], name = "input")
y = tf.placeholder(dtype = tf.int32, shape = [None], name = "labels")

with tf.name_scope("layer1"):
    layer1 = hiddenlayer(X, num_neurons1, activation = tf.nn.relu)
    output_layer1 = layer1.output()

with tf.name_scope("layer2"):
    layer2 = hiddenlayer(output_layer1, num_neurons2, activation = tf.nn.relu)
    output_layer2 = layer2.output()

with tf.name_scope("layer3"):
    layer3 = hiddenlayer(output_layer2, num_neurons3, activation = tf.nn.relu)
    output_layer3 = layer3.output()

with tf.name_scope("last_layer"):
    last_layer = hiddenlayer(output_layer3, classes)
    logits = last_layer.output()

with tf.name_scope("loss"):
    error = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(error)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("evaluate"):
    check_answer = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(check_answer, tf.float32))

def preprocess(data, labels = False):
    if labels:
        data = data.astype(np.int32)
    else:
        data = data.astype(np.float32)
        data /= 255.0
        data = data.reshape((-1, input_dsize))
    return data
    
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = [preprocess(d) for d in [x_train, x_test]]
y_train, y_test = [preprocess(d, labels = True) for d in [y_train, y_test]]
x_val, x_train = x_train[:5000], x_train[5000:]
y_val, y_train = y_train[:5000], y_train[5000:]

def get_batch(data, labels, batch_size):
    permutation = np.random.permutation(data.shape[0])
    index = permutation[:batch_size]
    return data[index], labels[index]

init = tf.global_variables_initializer()

iterations = x_train.shape[0] // batch_size

with tf.Session() as sess:
    init.run()
    for epoch in range(epochs):
        for iteration in range(iterations):
            x_batch, y_batch = get_batch(x_train, y_train, batch_size)
            sess.run(training_op, feed_dict = {X: x_batch, y: y_batch})
        accuracy_tmp = sess.run(accuracy, feed_dict = {X: x_val, y: y_val})
        print(epoch,":",accuracy_tmp)
    accuracy_final = sess.run(accuracy, feed_dict = {X: x_test, y: y_test})
    print("accuracy is", accuracy_final)
    



