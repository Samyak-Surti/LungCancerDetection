from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import tensorflow as tf
import os
import time
from threading import Timer

def printLayer(n):
    if n <= 12:
        print('Running layer ' + str(n) + '...')
        t = Timer(0.1, printLayer, args = [n+1])
        t.start()
    else:
        print('Running fully-connected layer...')

def load_graph(file_name): #Loads the old model's graph
    time.sleep(0.5)
    model_graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(file_name, 'rb') as f:
        graph_def.ParseFromString(f.read())
    with model_graph.as_default():
        tf.import_graph_def(graph_def)

    return model_graph


def read_tensor_from_image_file(file_name, input_height_param=299, input_width_param=299,
                                input_mean_param=0, input_std_param=255, channels_param=3):
    #Returns [1, 299, 299, 3] tensor from filname
    file_reader = tf.read_file(file_name, 'file_reader')
    if file_name.endswith('.png'): #Decode for each file type
        image_reader = tf.image.decode_png(file_reader, channels=channels_param,
                                           name='png_reader')
    elif file_name.endswith('.gif'):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=channels_param,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0) #Create graph to cast to floats, expand dimensions
    resized = tf.image.resize_bilinear(dims_expander, [input_height_param, input_width_param])
    normalized = tf.divide(tf.subtract(resized, [input_mean_param]), [input_std_param]) #Resize image and normalize
    session = tf.Session()
    result = session.run(normalized)
    return result

def read_tensor_from_image_file_pneumo(file_name, input_height_param=299, input_width_param=299,
                                input_mean_param=0, input_std_param=255, channels_param=1):
    #Returns [1, 299, 299, 3] tensor from filname
    file_reader = tf.read_file(file_name, 'file_reader')
    if file_name.endswith('.png'): #Decode for each file type
        image_reader = tf.image.decode_png(file_reader, channels=channels_param,
                                           name='png_reader')
    elif file_name.endswith('.gif'):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=channels_param,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0) #Create graph to cast to floats, expand dimensions
    resized = tf.image.resize_bilinear(dims_expander, [input_height_param, input_width_param])
    normalized = tf.divide(tf.subtract(resized, [input_mean_param]), [input_std_param]) #Resize image and normalize
    session = tf.Session()
    result = session.run(normalized)
    return result


def load_labels(file_name):
    fo = open(file_name, 'r')
    labels = fo.read()
    fo.close()
    return labels.split('\n')




if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors
    image_file = 'Validation/S. pyogenes/2.jpg'
    model = 'bacnet_v2.3'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Name of model (must be located in directory "Image labeling/Models"')
    parser.add_argument('--image', help='Image to be classified (can be any file type)')
    args = parser.parse_args()

    if args.model:
        model = args.model
    if args.image:
        image_file = args.image
