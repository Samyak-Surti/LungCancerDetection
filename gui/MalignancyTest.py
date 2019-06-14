import math
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops

from tensorflow.python.tools import inspect_checkpoint as chkp
from cnn_utils import *
import cv2
import h5py

# lets get the input tensor of the right size
m = 1 # number of test images
n_C = 1 # number of image channel
n_W = 448 # width of image in pixels
n_H = 448 # height of image in pixels


hdf5_path = '/home/samyaknn/ScienceProj2018/LungScan/lungdataset.hdf5'
subtract_mean = False

# open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")

# subtract the training mean
if subtract_mean:
    mm = hdf5_file["test_mean"][0, ...]
    mm = mm[np.newaxis, ...]

# Total number of samples
data_num = hdf5_file["test_img"].shape[0]
#print "There are", data_num, "test images."

from random import shuffle
from math import ceil
import matplotlib.pyplot as plt

batch_size = 2
nb_class = 3 # number of classes

from random import shuffle
from math import ceil
import matplotlib.pyplot as plt

# create list of batches to shuffle the data
batches_list = list(range(int(ceil(float(data_num) / batch_size))))
#print batches_list
shuffle(batches_list)
#print batches_list

# loop over batches
for n, i in enumerate(batches_list):
    i_s = i * batch_size  # index of the first image in this batch
    i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch

    # read batch images and remove training mean
    images = hdf5_file["test_img"][i_s:i_e, ...]
    if subtract_mean:
        images -= mm

    # read labels and convert to one hot encoding
    labels = hdf5_file["test_labels"][i_s:i_e]
    labels_one_hot = np.zeros((batch_size, nb_class))
    labels_one_hot[np.arange(batch_size), labels] = 1

    print n+1, '/', len(batches_list)

    #print 'Actual Diagnosis Label: ', labels[0], labels_one_hot[0, :]
    plt.imshow(images[0][:,:,-1], cmap='gray')
    plt.show()


    tf.reset_default_graph()

    sess=tf.Session()
   
    tf.global_variables_initializer()
   
    lung_model = tf.train.import_meta_graph("./Lung_Malignancy_Model.meta")
    lung_model.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name('X:0')	
    Y = graph.get_tensor_by_name('Y:0')
    final_tensor = graph.get_tensor_by_name('final_tensor:0')
    img = images[0]
    img= img[np.newaxis, :, :, :]
     
    y_test_images = np.zeros((1, 3))
   
    print 'Actual Diagnosis Label: ', labels_one_hot[0, :]
    predict_tensor = final_tensor.eval(feed_dict={X:img}, session=sess)
    print predict_tensor
    predict_tensor = predict_tensor[-1]
    print 'Predicted Probabilities: ', predict_tensor
    if np.dot(labels_one_hot[0, :].T, predict_tensor).all() == 0.:
	print 'WRONG PREDICTION!!'
    else:
	print 'Correct Prediction!!'
	

  
	  
  
