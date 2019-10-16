import h5py
import numpy as np
import matplotlib.pyplot as plt

hdf5_path = 'dataset.hdf5'
subtract_mean = False
# open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")
# subtract the training mean
if subtract_mean:
    mm = hdf5_file["train_mean"][0, ...]
    mm = mm[np.newaxis, ...]
# Total number of samples
train = hdf5_file["train_img"].shape
validate = hdf5_file["val_img"].shape
test = hdf5_file["test_img"].shape

print(train)
print(validate)
print(test)

print(type(hdf5_file["test_img"][1]))

print(hdf5_file["train_mean"])
