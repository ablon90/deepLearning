from keras_facile import *

#architecture du réseau
model = tf.keras.Sequential()
    #Couches de neurones
model.add(tf.keras.layers.Dense(2, input_dim=1, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='relu'))

model.summary()

#Poids de la couche 0
#Definir poids(model, couche, rang, coeff, biais)
definir_poids(model, 0, 0, 1, -1)
definir_poids(model, 0,1 ,-0.5, 1)
affiche_poids(model,0)

#Poids de la couche 0
definir_poids(model, 1, 0,[1,1], 0)
affiche_poids(model,1)

#Entrée/Sortie: une seule valeur
entree = np.array([[3.0]])
sortie = model.predict(entree)
print('\n'*10, '\nEntrée x = ',entree, '\nSortie F(x) = ',sortie)
