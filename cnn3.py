import os
import keras
from keras.models import Sequential 
from keras.layers.convolutional import Conv2D, MaxPooling2D 
from keras.layers.core import Dense, Flatten
from keras.datasets import mnist 
from keras.optimizers import Adam 

def networks(input_shape, num_outputs):
    model = Sequential()
    model.add(Conv2D(50, kernel_size = 5, input_shape = input_shape, activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(50, kernel_size = 5, activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation = "relu"))
    model.add(Dense(num_outputs, activation = "softmax"))
    return model

class MNISTData():
    def __init__(self):
        self.data_shape = (28, 28, 1)
        self.classes = 10 
    
    def get_batch(self):
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, labels = True) for d in [y_train, y_test]]
        return x_train, y_train, x_test, y_test 
    
    def preprocess(self, data, labels = False):
        if labels:
            data = keras.utils.to_categorical(data)
        else:
            data = data.astype("float32")
            data /= 255
            shape = (data.shape[0],) + self.data_shape
            data = data.reshape(shape)
        return data

class Trainer():
    def __init__(self, model, loss, optimizer):
        self._target = model 
        self._target.compile(loss = loss, optimizer = optimizer, metrics = ["accuracy"])
        self.verbose = 1
    
    def train(self, x_train, y_train, batch_size, epochs, validation_split):
        self._target.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_split = validation_split, verbose = self.verbose)

datasets = MNISTData()
x_train, y_train, x_test, y_test = datasets.get_batch()

model = networks(datasets.data_shape, datasets.classes)

trainer = Trainer(model, loss = "categorical_crossentropy", optimizer = Adam())
trainer.train(x_train, y_train, batch_size = 128, epochs = 12, validation_split = 0.2)

score = model.evaluate(x_test, y_test, verbose = 0)
print(score)


