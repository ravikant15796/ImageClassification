from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel,preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import random
from imutils import paths



IMAGE_SIZE = [200, 200] # IMAGE SIZE
batch_size = 128


train_path = 'data/train'
valid_path = 'data/test'


data_dir = os.path.join(os.getcwd(), 'data')
out_dir = os.path.join(os.getcwd(), 'output')


ptm = PretrainedModel(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False)

ptm.trainable = False


K = 2 # number of classes
x = Flatten()(ptm.output)
x = Dense(K, activation='softmax')(x)

model = Model(inputs=ptm.input, outputs=x)
model.summary()


gen_train = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  preprocessing_function=preprocess_input
)

gen_test = ImageDataGenerator(
  preprocessing_function=preprocess_input
)


# create generators
train_generator = gen_train.flow_from_directory(
  train_path,
  shuffle=True,
  target_size=IMAGE_SIZE,
  batch_size=batch_size,
)
valid_generator = gen_test.flow_from_directory(
  valid_path,
  shuffle = False,
  target_size=IMAGE_SIZE,
  batch_size=batch_size,
)


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(valid_path + '/*/*.jpg')


r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=10,
  steps_per_epoch=int(np.ceil(len(image_files) / batch_size)),
  validation_steps=int(np.ceil(len(valid_image_files) / batch_size)),
)


plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()


plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


model_path = os.path.join(out_dir,  'trained_VGG_model.h5')
model.save(model_path)

predIdxs=model.predict_generator(valid_generator)
predIdxs = np.argmax(predIdxs, axis=1)

print('Confusion Matrix')
print(confusion_matrix(valid_generator.labels, predIdxs))

print(classification_report(valid_generator.labels, predIdxs,
                            target_names=["apple", "non-apple"]))