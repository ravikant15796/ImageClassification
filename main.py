import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import random
from imutils import paths

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel,preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input



test_data_dir = os.path.join(os.getcwd(), 'test')
out_dir = os.path.join(os.getcwd(), 'output')




model_path = os.path.join(out_dir,  'trained_VGG_model.h5')
model = keras.models.load_model(model_path)

# load an images from test folder
imagePaths = sorted(list(paths.list_images(test_data_dir)))

for imgPath in imagePaths:
    label = ""
    image = load_img(imgPath, target_size=(200, 200))
   
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # prepare the image for the VGG model
    image = preprocess_input(image)
    yhat = model.predict(image)
    if np.ceil(yhat)[0][0] == np.ceil(yhat)[0][1]:
        label = "Not apple"
    elif np.ceil(yhat)[0][0] == 1.:
        label = "apple"
    else:
        label = "Not apple"

    print(imgPath,'----',label)