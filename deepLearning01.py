import tensorflow as tf
import numpy as np

#architecture du réseau
model = tf.keras.Sequential()

#Couches de neurones
model.add(tf.keras.layers.Dense(2, input_dim=1, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='relu'))

#Couche 0 - Définir à la main les poids
coeff = np.array([[1.,-0.5]])
biais = np.array([-1, 1])
poids = [coeff, biais]
model.layers[0].set_weights(poids)

#Couche 1 - Définir à la main les poids
coeff = np.array([[1.0] , [1.0]])
biais = np.array([0])
poids = [coeff, biais]
model.layers[1].set_weights(poids)

#Entrée/Sortie: une seule valeur
entree = np.array([[3.0]])
sortie = model.predict(entree)
print('\n'*10, '\nEntrée x = ',entree, '\nSortie F(x) = ',sortie)
