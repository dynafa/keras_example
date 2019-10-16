import h5py

file = h5py.File("Ass_1_Part_3/datasets/test_catvnoncat.h5", "r")
file.flush()
file.close()

