#!/home/minami/tf2.0/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import h5py
import numpy as np

hdf5_path = 'dataset.hdf5'
subtract_mean = False
# open the hdf5 file
df = h5py.File(hdf5_path, "r")
# subtract the training mean
if subtract_mean:
    mm = df["train_mean"][0, ...]
    mm = mm[np.newaxis, ...]
# Total number of samples
train = df["train_img"].shape
validate = df["val_img"].shape
test = df["test_img"].shape

print(train)
print(validate)
print(test)

modelname = 'DNN_cat_dog_model.h5'
newmodelname = 'DNN_cat_dog_improved.h5'
epochs = 100
dims_X = 100
dims_Y = 100
train_images, train_labels, test_images, test_labels = \
    df["train_img"], df["train_labels"], df["test_img"], df["test_labels"]

train_images = train_images[:train[0]].reshape(-1, dims_X * dims_Y, 3) / 255.0
test_images = test_images[:test[0]].reshape(-1, dims_X * dims_Y, 3) / 255.0
train_labels = train_labels[:15000]
test_labels = test_labels[:5000]



# Recreate the exact same model, including its weights and the optimizer
saved_model = tf.keras.models.load_model(modelname)

# Show the model architecture
saved_model.summary()

# Re-evaluate the model
loss, acc = saved_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print(saved_model.optimizer.get_config())

# To get learning rate
# To set learning rate
K.set_value(saved_model.optimizer.lr, 0.001)
K.set_value(saved_model.optimizer.decay, 0.01)
K.set_value(saved_model.optimizer.momentum, 0.9)
print(K.get_value(saved_model.optimizer.lr))
print(K.get_value(saved_model.optimizer.decay))
print(K.get_value(saved_model.optimizer.momentum))

# print(saved_model.optimizer.get_config())

input("Ready?")
saved_model.fit(train_images, train_labels, epochs=epochs)

saved_model.save(newmodelname)

print("Model saved as %s" % newmodelname)
print("Completed training network for %s epochs" % epochs)
loss, acc = saved_model.evaluate(test_images, test_labels, verbose=0)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))
print(loss)
