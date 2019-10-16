import numpy as np
import h5py

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow

# check the order of data and chose proper data shape to save images
if data_order == 'th':
    train_shape = (len(train_addrs), 3, 224, 224)
    val_shape = (len(val_addrs), 3, 224, 224)
    test_shape = (len(test_addrs), 3, 224, 224)
elif data_order == 'tf':
    train_shape = (len(train_addrs), 224, 224, 3)
    val_shape = (len(val_addrs), 224, 224, 3)
    test_shape = (len(test_addrs), 224, 224, 3)

# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("val_img", val_shape, np.int8)
hdf5_file.create_dataset("test_img", test_shape, np.int8)

hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("val_labels", (len(val_addrs),), np.int8)
hdf5_file["val_labels"][...] = val_labels
hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.int8)
hdf5_file["test_labels"][...] = test_labels