import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

tf.debugging.set_log_device_placement(True)

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
epochs = 50
dims_X = 100
dims_Y = 100
train_images, train_labels, test_images, test_labels = \
    df["train_img"], df["train_labels"], df["test_img"], df["test_labels"]

# train_images = train_images[:train[0]].reshape(-1, 28 * 28) / 255.0
# test_images = test_images[:test[0]].reshape(-1, 28 * 28) / 255.0
# #
# train_images = train_images[:train[0]] / 255.0
# test_images = test_images[:test[0]] / 255.0
train_images = train_images[:train[0]].reshape(-1, dims_X * dims_Y, 3) / 255.0
test_images = test_images[:test[0]].reshape(-1, dims_X * dims_Y, 3) / 255.0
train_labels = train_labels[:15000]
test_labels = test_labels[:5000]
sdg = keras.optimizers.SGD(learning_rate=0.01, momentum=0, decay=0, nesterov=False)


# Define a simple sequential model
def create_model():
    mymodel = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(dims_X * dims_Y, 3)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation='softmax')
    ])
    mymodel.compile(optimizer=sdg,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return mymodel


print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
print(type(train_images), type(train_labels), type(test_images), type(test_labels))

# print(sdg)
# Create and train a new model instance.
model = create_model()
print(model.summary())
print(model.optimizer.get_config())
print("Epochs: %d" % epochs)

input("Ready?")
model.fit(train_images, train_labels, epochs=epochs)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model shuold be saved to HDF5.
model.save(modelname)

print("Model saved as %s" % modelname)
print("Completed training network for %s" % epochs)
loss, acc = model.evaluate(test_images, test_labels)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))
print(loss)
