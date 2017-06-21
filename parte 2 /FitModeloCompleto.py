#librerias utilitarias
from random import randint
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

#librerias Keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad

train_data = sio.loadmat('data/extra_32x32.mat')
test_data = sio.loadmat('data/test_32x32.mat')
X_train = train_data['X'].T
y_train = train_data['y'] - 1
X_test = test_data['X'].T
y_test = test_data['y'] - 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
n_classes = len(np.unique(y_train))
print (np.unique(y_train))

from keras.utils import np_utils
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print (X_train.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_test.shape)

from keras import backend as K
K.image_data_format()

#Parametros
_size, n_channels, n_rows, n_cols = X_train.shape
adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)

model = Sequential()
model.add(Convolution2D(16, 5, 5, border_mode='same', activation='relu',
                        input_shape=(n_channels, n_rows, n_cols)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(512, 7, 7, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=adagrad,  metrics=['accuracy'])

adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
model.fit(X_train, Y_train, batch_size=1280, epochs=12, verbose=1, \
          validation_data=(X_test, Y_test))

model.save('modeloCompleto.h5')