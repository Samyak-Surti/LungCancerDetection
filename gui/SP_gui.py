import random
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy
from Tkinter import *
from ttk import *
from PIL import Image, ImageTk
from inception_label import *
import numpy as np
import tensorflow as tf
from scipy import ndimage
from random import shuffle
from math import ceil
from OneIterMalignancyTest import *
from tensorflow.python.tools import inspect_checkpoint as chkp
from cnn_utils import *
import cv2
import h5py

imageDirectory = 'Images'
modelsDirectory = "Models"
dataset_path = '/home/samyaknn/ScienceProj2018/LungScan/lungdataset.hdf5'


class App: #Contains all the widgets and neccesary functions
    def __init__(self, master):
	'''
	Sets up all the tkinter widgets
	Inputs: A tk master
	Outputs: None
	'''        
	self.master = master
        self.frame = Frame(master)

        self.correctClassification = '' #Show CT scan
        self.imagePath = ''
        self.imageWidget = Label(self.frame)
        self.imageWidget.pack(padx=10, pady=5, side=LEFT)

        self.controlFrame = Frame(self.frame) #Holds all the "controls" (everything but image)
        self.buttonFrame = Frame(self.controlFrame) #Holds two buttons
        self.modelFrame = Frame(self.controlFrame) #Holds the text "model: " and dropbox

        self.nextButton = Button(self.buttonFrame, text='New Image', command=self.next_image)
        self.nextButton.pack(side=LEFT, padx=5)
	
	self.classifyButton = Button(self.buttonFrame, text = 'Classify Image', command = self.classify)
	self.classifyButton.pack(side = LEFT, padx = 5)

	#self.pneumoButton = Checkbutton(self.buttonFrame, text = 'Pneumonet')
	#self.pneumoButton.pack(side = LEFT, padx = 5)

	#self.inceptionButton = Checkbutton(self.buttonFrame, text = 'Inception')
	#self.inceptionButton.pack(side = LEFT, padx = 5)

        self.buttonFrame.pack(pady=5)

        self.modelLabel = Label(self.modelFrame, text='Using model "Pneumonet"')
        self.modelLabel.pack(padx=5, pady=5)
	
        self.modelFrame.pack()

        self.outputText = [] #Create a blank output area
        self.outputLabel = Label(self.controlFrame, width=45)
        self.outputLabel.pack(padx=10, pady=5, side=LEFT)
        self.outputHeight = 40

        self.controlFrame.pack(side=TOP, padx=10, pady=10)

        self.next_image() #Find the first image to display
	self.classify()

        self.frame.pack()

    def clear_screen(self):
	#Clears the output area 
        self.outputText = []
        self.outputLabel.config(text=self.outputHeight*'\n')

    def classify(self):
	#Performs the classification of the loaded image
        classificationModel = 'Pneumonet'
        #classificationModelCompare = 'Inception'
        self.label_print('')
        self.label_print('Classifying image using model "' + classificationModel + '"...')
        if classificationModel == 'Pneumonet':
            probabilities = self.classify_pneumo(self.imagePath,dataset_path)
        else:
	    probabilities = self.classify_inception(self.imagePath)   
	self.label_print('Computed probabilities:')
        for category in probabilities:
            self.label_print('  ' + category)

    def next_image(self):
	#Picks a random image, writes the class string to correctClassification, and updates the image widget
        self.correctClassification = random.choice(os.listdir(imageDirectory))
	correctPath = os.path.join(imageDirectory, self.correctClassification)
        self.imagePath = os.path.join(correctPath, random.choice(os.listdir(correctPath)))

        image = Image.open(self.imagePath)
	image = image.resize((1000,1000), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)

        self.imageWidget.config(image=photo)
        self.imageWidget.image = photo # keep a reference!

        self.clear_screen()
        self.label_print('New image loaded')
        self.label_print('Correction classification: ' + self.correctClassification)

    def label_print(self, text):
	#Prints to the output label, has support for scrolling (prints one line at a time)
        self.outputText.append(text + '\n')
        outputLength = len(self.outputText)
        if outputLength > self.outputHeight:
            self.outputText.pop(0)
            self.outputLabel.config(text=''.join(self.outputText))
        else:
            self.outputLabel.config(text=''.join(self.outputText) + (self.outputHeight-outputLength)*'\n')
        self.outputLabel.update()

    def classify_inception(self, image):
	#Performs classification for the "inception" network
        input_layer = 'Mul'  # final_training_ops
        output_layer = 'final_result'

        model_file = 'Models/Inception' + '/output_graph.pb' #Load appropriate files for model
        label_file = 'Models/Inception' + '/output_labels.txt'
        self.label_print('Loading model graph...')
        graph = load_graph(model_file)

        self.label_print('Reading tensor from image file...')
        t = read_tensor_from_image_file(image,
                                        input_height_param=299,
                                        input_width_param=299,
                                        input_mean_param=0,
                                        input_std_param=255)

        input_name = 'import/' + input_layer
        output_name = 'import/' + output_layer
        input_operation = graph.get_operation_by_name(input_name) #Name the graph files
        output_operation = graph.get_operation_by_name(output_name)

        self.label_print('Running model graph...')
        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0],#Run inception graph, find results
                               {input_operation.outputs[0]: t})

        results = np.squeeze(results)

        top_k = results.argsort()[::-1]
        labels = load_labels(label_file) #Create output based on label file
        output = ''
        for i in top_k:
            output += labels[i] + ': '
            output += '%.1f' % (results[i] * 100) + '%' + '\n'

        self.label_print('Finished!')
        self.label_print('')
        return output.split('\n')

    def classify_pneumo(self, image, dataset_path):
	# Will take random image to train on

	self.label_print('Reading tensor from image file...')
	one_hot_labels = PickRandomImage(dataset_path)
	
	tf.reset_default_graph()

	self.label_print('Running model graph...')

	sess = tf.Session()
	tf.global_variables_initializer()

	lung_model = tf.train.import_meta_graph("./Lung_Malignancy_Model.meta")
	lung_model.restore(sess, tf.train.latest_checkpoint('./'))
	graph = tf.get_default_graph()
	
	X = graph.get_tensor_by_name('X:0')	
        Y = graph.get_tensor_by_name('Y:0')
	final_tensor = graph.get_tensor_by_name('final_tensor:0')
	img = image[0]
	img = img[np.newaxis, :, :, :]

	y_test_images = np.zeros((1,3))

	self.label_print('Actual Diagnosis Label: ', one_hot_labels[0, :])
	predict_tensor = final_tensor.eval(feed_dict = {x:img}, session = sess)
	predict_tensor = predict_tensor[-1]
	self.label_print('Predicted Probabilities: ', predict_tensor)
	if np.dot(one_hot_labels[0, :].T, predict_tensor).all() == 0.:
	    self.label_print ('WRONG PREDICTION!!')
	else:
	    self.label_print('Correct Prediction!!')

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Create app
    root = Tk()
    root.wm_title('Convolutional Neural Network Lung Cancer Diagnosis Tool')
    App(root)
    root.mainloop()
    try:
        root.destroy()
    except TclError:
        pass

if __name__ == '__main__':
    main()
