import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
### Partie A - Création des données
from keras.datasets import mnist
from keras.utils import to_categorical
(X_train_data,Y_train_data),(X_test_data,Y_test_data) = mnist.load_data()
N = X_train_data.shape[0] # 60 000 données

X_train = np.reshape(X_train_data, (N,28,28,1))
X_test = np.reshape(X_test_data, (X_test_data.shape[0],28,28,1))
X_train = X_train/255 # normalisation
X_test = X_test/255
Y_train = to_categorical(Y_train_data, num_classes=10)
Y_test = to_categorical(Y_test_data, num_classes=10)


modele02 = keras.models.load_model('modele_Save')

score = modele02.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])