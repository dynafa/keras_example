#!/home/minami/tf2.0/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print(train)
print(validate)
print(test)

model_1_name = 'VGG_cat_dog_model.h5'
model_2_name = 'cat_dog_model.h5'
dims_X = 100
dims_Y = 100
train_images_orig, train_labels, test_images_orig, test_labels = \
    df["train_img"], df["train_labels"], df["test_img"], df["test_labels"]

train_images = train_images_orig[:train[0]] / 255.0
test_images = test_images_orig[:test[0]] / 255.0
train_labels = train_labels[:15000]
test_labels = test_labels[:5000]

train_images_flat = train_images[:train[0]].reshape(-1, dims_X * dims_Y, 3) / 255.0
test_images_flat = test_images[:test[0]].reshape(-1, dims_X * dims_Y, 3) / 255.0

# Recreate the exact same model, including its weights and the optimizer
saved_model_1 = tf.keras.models.load_model(model_1_name)
saved_model_2 = tf.keras.models.load_model(model_2_name)

start = time.time()

predictions_1 = saved_model_1.predict(test_images[:100])
predictions_2 = saved_model_2.predict(test_images_flat[:100])




# plt.subplots_adjust(left=0.05)
# plt.subplots_adjust(right=0.95)
# plt.subplots_adjust(top=0.95)
# plt.subplots_adjust(bottom=0.05)
# plt.subplots_adjust(wspace=0.05)
# plt.subplots_adjust(hspace=0.05)

# for x in range(0, 100):
#     print(predictions_1[x])
#     plt.imshow(test_images_orig[x])
#     plt.show()
for x in range(1, 41, 2):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplot(5, 8, x)
    plt.plot(predictions_1[x])
    # plt.plot(test_images_orig[x][:, :, 0].flatten()/255, color="red", linewidth=0.5)
    # plt.plot(test_images_orig[x][:, :, 1].flatten()/255, color="green", linewidth=0.3)
    # plt.plot(test_images_orig[x][:, :, 2].flatten()/255, color="blue", linewidth=0.1)
for x in range(2, 41, 2):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplot(5, 8, x)
    plt.plot(predictions_2[x])
try:
    plt.savefig("001.png", dpi=300)
except KeyboardInterrupt:
    sys.exit(0)
