import random
from math import ceil
import matplotlib.pyplot as plt
from Tkinter import *
from ttk import *
from PIL import Image, ImageTk
from inception_label import *
import numpy as np
import h5py

datasetDirectory = 'LungScan/lungdataset.hdf5'
modelsDirectory = "Models"


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
        classificationModel = 'Inception'
        self.label_print('')
        self.label_print('Classifying image using model "' + classificationModel + '"...')
        if classificationModel == 'Inception':
            probabilites = self.classify_inception(self.imagePath)
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

    def next_image(self):
	subtract_mean = False
	
	# Open the hdf5 file
	hdf5_file = h5py.File(hdf5_path, "r")
	
	# Subtract the training mean
	if subtract_mean:
	    mm = hdf5_file["test_mean"][0, ...]
   	    mm = mm[np.newaxis, ...]

	# Number of images in test dataset
	data_num = hdf5_file["test_img"].shape[0]

	batch_size = 2
	nb_class = 3
	
	# Create list of batches to shuffle the data
	batches_list = list(range(int(ceil(float(data_num) / batch_size))))
	# Shuffle batches list
	shuffle(batches_list)

	# Loop over batches
	for n, i in enumerate(batches_list):
	    i_s = i * batch_size # Gives index of first image in current batch
	    i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in current batch

	    images = hdf5_file["test_img"][i_s:i_e, ...]
	    if subtract_mean:
		images -= mm
	
	

	

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

	#def classify_pneumonet(self,


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
