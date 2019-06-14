from random import shuffle
import glob
shuffle_data = True  # shuffle the addresses before saving
hdf5_path = '/home/samyaknn/ScienceProj2018/LungScan/lungdataset.hdf5'  # address to where you want to save the hdf5 file
lungscan_train_path = '/home/samyaknn/ScienceProj2018/LungScan/*/*.jpg'

train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

n_C = 1 # number of image channel
n_W = 448 # width of image in pixels
n_H = 448 # height of image in pixels

# read addresses and labels from the 'train' folder
addrs = glob.glob(lungscan_train_path)
#print len(addrs)

labels = []
for addr in addrs:
   #print(addr)
   if 'Benign' in addr:
	label = 0
   elif 'Malignant' in addr:
	label = 1
   else:
	label = 2
   labels.append(label)
   

# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
    
# Divide the hata into  train, validation, and test fractions
train_addrs = addrs[0:int(train_frac*len(addrs))]
train_labels = labels[0:int(train_frac*len(labels))]

val_addrs = addrs[int(train_frac*len(addrs)):int((train_frac+val_frac)*len(addrs))]
val_labels = labels[int(train_frac*len(addrs)):int((train_frac+val_frac)*len(addrs))]

test_addrs = addrs[int((train_frac+val_frac)*len(addrs)):]
test_labels = labels[int((train_frac+val_frac)*len(labels)):]

import numpy as np
import h5py
import cv2

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow

# check the order of data and chose proper data shape to save images
if data_order == 'th':
    train_shape = (len(train_addrs), n_C, n_W, n_H)
    val_shape = (len(val_addrs), n_C, n_W, n_H)
    test_shape = (len(test_addrs), n_C, n_W, n_H)
elif data_order == 'tf':
    train_shape = (len(train_addrs), n_W, n_H, n_C)
    #print train_shape
    val_shape = (len(val_addrs), n_W, n_H, n_C)
    #print val_shape
    test_shape = (len(test_addrs), n_W, n_H, n_C)
    #print test_shape

# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("train_img", train_shape, np.int16)
hdf5_file.create_dataset("val_img", val_shape, np.int16)
hdf5_file.create_dataset("test_img", test_shape, np.int16)

hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("val_labels", (len(val_addrs),), np.int8)
hdf5_file["val_labels"][...] = val_labels
hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.int8)
hdf5_file["test_labels"][...] = test_labels

# a numpy array to save the mean of the images
mean = np.zeros(train_shape[1:], np.float32)

# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print 'Train data: {}/{}'.format(i, len(train_addrs))

    # read an image and resize to (n_W, n_H)
    # cv2 load images as BGR, convert it to GRAY
    addr = train_addrs[i]
    print(addr)
    img = cv2.imread(addr)
    img = cv2.resize(img, (n_W, n_H), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:, :, np.newaxis]
    #print(img.shape)

    # add any image pre-processing here

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))

# loop over validation addresses
for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print 'Validation data: {}/{}'.format(i, len(val_addrs))

    # read an image and resize to (n_W, n_H)
    # cv2 load images as BGR, convert it to GRAY
    addr = val_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (n_W, n_H), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:, :, np.newaxis]

    # add any image pre-processing here

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    # save the image
    hdf5_file["val_img"][i, ...] = img[None]

# loop over test addresses
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print 'Test data: {}/{}'.format(i, len(test_addrs))

    # read an image and resize to (n_W, n_H)
    # cv2 load images as BGR, convert it to GRAY
    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (n_W, n_H), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:, :, np.newaxis]


    # add any image pre-processing here

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    # save the image
    hdf5_file["test_img"][i, ...] = img[None]

# save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()

