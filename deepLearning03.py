import pathlib

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


modele02 = keras.models.load_model('modele_Save')
print(modele02.summary())

import PIL
from PIL import Image

#fonction return result of the model
def res(x):
    tab = np.array(x)
    maxi = np.max(tab)
    _,index = np.where(tab == maxi)
    return (int(index), maxi)

    #generalisation

#transform stock images into exploitable input for the model
def transformation_input(image, shape):

    image_grey = image.convert("L")
    image_grey_array = np.array(image_grey) / 255
    image_grey_array_reshape = image_grey_array.reshape(shape)
    return image_grey_array_reshape

#========================
#       variables
#======================
path_directory = "directory"
num = []
shape = (-1, 28, 28, 1)
images = []
    #results of the model
res_model = []
verif = []

# stock images from path_directory
for path in pathlib.Path(path_directory).iterdir():
    if path.is_file():
        images.append(PIL.Image.open(path))
        num.append(str(path)[len(path_directory + "\\")])
image_transformation_input = np.array([transformation_input(_, shape) for _ in images])



for i in image_transformation_input:
    res_model.append(modele02(i, training="false"))

score = np.array( [res(_) for _ in res_model] )
for i in range(len(score)):
    if int(num[i]) == int(score[i][0]):
        verif.append("OK")
    else:
        verif.append("NOP")


# function that display the results
def output(num, score):
    first_ligne = ["Predicted number", " Probability"]
    ligne = len(str(num[0])) + 3*len(3*"    ") + len(str(score[0][0]))  + len(str(score[0][1])) + len("NOP")
    print("=" * ligne)
    print("Results")
    print("-" * ligne)
    print(1*"\t", end="")
    for i in first_ligne:
        print(i, end="\t")
    print("\n" + "-"*ligne)

    k=0
    for i in score:
        print(num[k], end=3*"\t")
        print(int(i[0]), 3*"\t", i[1], 3*"\t", verif[k])
        k = k + 1
    print("-" * ligne)
    print("Corrects :", verif.count("OK")/len(verif))
    print("False :", verif.count("NOP")/len(verif))
    print("=" * ligne)

output(num, score)


