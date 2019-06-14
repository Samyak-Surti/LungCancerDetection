from Tkinter import *
#from Tkinter.ttk import *
from PIL import Image, ImageTk
from threading import Timer


class App:
    def __init__(self, master):
        self.master = master  #Setup tk stuff
        self.frame = Frame(master)

        #self.topLabel = Label(self.frame, text='Cancer_net_v2')
        #self.topLabel.pack(pady=5)

        image = Image.open("pathology.jpg")
        photo = ImageTk.PhotoImage(image)

        self.imageWidget = Label(self.frame, image=photo)
        self.imageWidget.image = photo # keep a reference!
        self.imageWidget.pack(padx=10, pady=5, side=LEFT)

        self.controlFrame = Frame(self.frame)
        self.buttonFrame = Frame(self.controlFrame)

        self.classifyButton = Button(self.buttonFrame, text='Classify Image', command=self.classify)
        self.classifyButton.pack(side=LEFT, padx=5)

        self.nextButton = Button(self.buttonFrame, text='Next Image', command=self.next)
        self.nextButton.pack(side=LEFT, padx=5)

        self.buttonFrame.pack(pady=5)

        self.progressBar = Progressbar(self.controlFrame, length=200)
        self.progressBar.pack()

        self.outputText = []
        self.outputLabel = Label(self.controlFrame)
        self.outputLabel.pack(padx=10, pady=5, side=LEFT)
        self.outputHeight = 24

        self.controlFrame.pack(side=TOP, padx=10, pady=10)

        self.label_print('Neural network ready')
        self.label_print('')

        self.frame.pack()

    def print_results(self):
        self.progressBar.stop()
        self.label_print('Probabilities:')
        self.label_print('  Class 1: 97%')
        self.label_print('  Class 2: 2%')
        self.label_print('  Class 3: 1%')
        self.label_print('')

    def clear_screen(self):
        self.outputText = []
        self.outputLabel.config(text=self.outputHeight*'\n')

    def classify(self):
        self.label_print('Classifying...')
        self.progressBar.start(40)
        t = Timer(4, self.print_results)
        t.start()

    def next(self):
        self.label_print('Next button pressed')

    def label_print(self, text):
        self.outputText.append(text + '\n')
        outputLength = len(self.outputText)
        if outputLength > self.outputHeight:
            self.outputText.pop(0)
            self.outputLabel.config(text=''.join(self.outputText))
        else:
            self.outputLabel.config(text=''.join(self.outputText) + (self.outputHeight-outputLength)*'\n')


def main():
    root = Tk()
    root.wm_title('Cancer_net_v2')
    app = App(root)
    root.mainloop()
    try:
        root.destroy()
    except TclError:
        pass

if __name__ == '__main__':
    main()
