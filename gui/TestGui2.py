import random
from math import ceil
import matplotlib.pyplot as plt
from Tkinter import *
from ttk import *
from PIL import Image, ImageTk
from inception_label import *
import numpy as np
import tensorflow as tf
import h5py

imageDirectory = 'Images'
modelsDirectory = 'Models'


class App:
    def __init__(self, master):
        self.master = master  #Setup tk stuff
        self.frame = Frame(master)

        self.correctClassification = ''
        self.imagePath = ''
        self.imageWidget = Label(self.frame)
        self.imageWidget.pack(padx=10, pady=5, side=LEFT)

        self.controlFrame = Frame(self.frame)
        self.buttonFrame = Frame(self.controlFrame)
        self.modelFrame =Frame(self.controlFrame)

        self.classifyButton = Button(self.buttonFrame, text='Classify Image', command=self.classify)
        self.classifyButton.pack(side=LEFT, padx=5)

        self.nextButton = Button(self.buttonFrame, text='New Image', command=self.next_image)
        self.nextButton.pack(side=LEFT, padx=5)

        self.buttonFrame.pack(pady=5)

        self.model = StringVar(self.master)
        self.model.set('Pneumo_net') # default value

        self.modelLabel = Label(self.modelFrame, text='Model: ')
        self.modelLabel.pack(side=LEFT, padx=5)

        self.modelMenu = OptionMenu(self.modelFrame, self.model, '', 'Pneumo_net', 'Inception')
        self.modelMenu.pack(side=LEFT, padx=5)

        self.modelFrame.pack()

        self.outputText = []
        self.outputLabel = Label(self.controlFrame, width=45)
        self.outputLabel.pack(padx=10, pady=5, side=LEFT)
        self.outputHeight = 28

        self.controlFrame.pack(side=TOP, padx=10, pady=10)

        self.next_image()

        self.frame.pack()

    def clear_screen(self):
        self.outputText = []
        self.outputLabel.config(text=self.outputHeight*'\n')

    def classify(self):
        classificationModel = 'Pneumonet'
        self.label_print('')
        self.label_print('Classifying image using model "' + classificationModel + '"...')
        if classificationModel == 'Pneumonet':
            probabilites = self.classify_pneumonet(self.imagePath)
            self.label_print('Computed probabilites:')
            for category in probabilites:
                self.label_print('  ' + category)

    def next_image(self):
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
        self.outputText.append(text + '\n')
        outputLength = len(self.outputText)
        if outputLength > self.outputHeight:
            self.outputText.pop(0)
            self.outputLabel.config(text=''.join(self.outputText))
        else:
            self.outputLabel.config(text=''.join(self.outputText) + (self.outputHeight-outputLength)*'\n')
        self.outputLabel.update()

    def classify_inception(self, image):
        input_layer = 'Mul'  # final_training_ops
        output_layer = 'final_result'

        model_file = 'Models/Inception' + '/output_graph.pb'
        label_file = 'Models/Inception' + '/output_labels.txt'
        self.label_print('Loading model graph...')
        graph = load_graph(model_file)

        self.label_print('Reading tensor from image file...')
        t = read_tensor_from_image_file_pneumo(image,
                                        input_height_param=299,
                                        input_width_param=299,
                                        input_mean_param=0,
                                        input_std_param=255)

        input_name = 'import/' + input_layer
        output_name = 'import/' + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        self.label_print('Running model graph...')
        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: t})

        results = np.squeeze(results)

        top_k = results.argsort()[::-1]
        labels = load_labels(label_file)
        output = ''
        for i in top_k:
            output += labels[i] + ': '
            output += '%.1f' % (results[i] * 100) + '%' + '\n'

        self.label_print('Finished!')
        self.label_print('')
        return output.split('\n')

    def classify_pneumonet(self, image):
	label_file = 'Models/Inception' + '/output_labels.txt'
	
	self.label_print('Loading model graph...')


	tf.reset_default_graph()
	sess = tf.Session()
	tf.global_variables_initializer()

	lung_model = tf.train.import_meta_graph("./Lung_Malignancy_Model.meta")
	lung_model.restore(sess,tf.train.latest_checkpoint('./'))	

	graph = tf.get_default_graph()

	X = graph.get_tensor_by_name('X:0')
	Y = graph.get_tensor_by_name('Y:0')
	final_tensor = graph.get_tensor_by_name('final_tensor:0')
	self.label_print('Reading tensor from image file...')
	img = image[0]
	img = read_tensor_from_image_file(image,
					  input_height_param = 448,
					  input_width_param = 448,
				  	  input_mean_param = 0,
					  input_std_param = 255)
	tf.image.rgb_to_grayscale(img)
	
	y_test_images = np.zeros((1,3))

	labels = load_labels(label_file)
	output = ''
	self.label_print('Running model graph...')
	#print 'Actual Diagnosis Label', labels[0, :]
	predict_tensor = final_tensor.eval(feed_dict = {X:img}, session = sess)
	predict_tensor = np.squeeze(predict_tensor)
	print predict_tensor
	top_k = predict_tensor.argsort()[::-1]
	for i in top_k:
	    output += labels[i] + ': '
	    output += '%.1f' % (predict_tensor[i] * 100) + '%' + '\n'

	self.label_print('Finished!')
	self.label_print('')
	return output.split('\n')


	
	

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
