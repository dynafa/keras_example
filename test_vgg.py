import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print(physical_devices)

input("STOP")

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

modelname = 'VGG_cat_dog_model.h5'
epochs = 20
dims_X = 100
dims_Y = 100
train_images, train_labels, test_images, test_labels = \
    df["train_img"], df["train_labels"], df["test_img"], df["test_labels"]

# train_images = train_images[:train[0]].reshape(-1, 28 * 28) / 255.0
# test_images = test_images[:test[0]].reshape(-1, 28 * 28) / 255.0
# #
train_images = train_images[:train[0]] / 255.0
test_images = test_images[:test[0]] / 255.0
# train_images = train_images[:train[0]].reshape(-1, 112 * 112, 3) / 255.0
# test_images = test_images[:test[0]].reshape(-1, 112 * 112, 3) / 255.0
train_labels = train_labels[:15000]
test_labels = test_labels[:5000]

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
print(type(train_images), type(train_labels), type(test_images), type(test_labels))

# print(sdg)
# Create and train a new model instance.
model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

epochs = eval(input("How many epochs?: "))
print("Epochs: %d" % epochs)

input("Ready?")
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size=32, epochs=10)
model.fit(test_images, test_labels, batch_size=32, epochs=epochs)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model shuold be saved to HDF5.
model.save(modelname)

print("Model saved as %s" % modelname)
print("Completed training network for %s" % epochs)
loss, acc = model.evaluate(test_images, test_labels, batch_size=32)
# loss, acc = model.evaluate(x_test, y_test, batch_size=32)

print("Trained model, accuracy: {:5.2f}%".format(100*acc))
