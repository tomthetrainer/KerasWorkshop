'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


# Format console print to see image as 28 * 28
np.core.arrayprint._line_width = 320

batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

'''
Returns:

2 tuples:
    x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
    y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).
'''

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print("Array of first digit pixel values")
print(X_train[1])
print("Label")
print(y_train[1])

testimage = X_train[1]
#X = X.reshape(1,784)
testimage = testimage.reshape(1,784)
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Label is now 1-hot encoded
print(Y_train[1])

model = Sequential()
# add a dense layer of 512 nodes
model.add(Dense(512, input_shape=(784,)))
# set Activation to relu
model.add(Activation('relu'))
# set dropout to .2
model.add(Dropout(0.2))
# Add another dense layer same values as first
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# add output layer of 10 nodes for our classification
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])




'''
RMSprop divides the learning rate by an exponentially
decaying average of squared gradients.
'''


history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# Get some predictions
prediction = model.predict(X_train[0].reshape(1,784));
print(prediction)
print(Y_train[0])
prediction = model.predict(X_train[1].reshape(1,784));
print(prediction)
print(Y_train[1])

# Save the trained model
model.save('mnist_mlp.h5')

