# A Deep Learning Based Approach To Improve Accuracy in Lung Cancer Malignancy Detection #

## Abstract ##
Lung cancer is a life-threatening disease that affects over 225,000 Americans per year. The most common test used to screen for lung cancer is the Computed Tomography (CT) scan which provides a slice-by-slice representation of our lung cavity. Radiologists will have to go through these CT scan images by hand to spot any nodules that may be present in the lung cavity. Currently, performing an accurate diagnosis is a very lengthy procedure comprising of a biopsy and many other tests. These tests can be avoided in the future by improving accuracy using deep learning-based approaches. We proposed a deep learning method that would give a highly accurate diagnosis of the patient’s lung cancer malignancy purely based on the images from their CT scan. The initial approach was to use a purpose-built 4-layer 2D CNN to classify the CT scan images for one of three classes: benign, malignant, or metastatic. We began by scaling all of our images to 448 x 448 to retain as much information as possible as it went through each hidden layer. Our next approach employed the use of a slightly bigger 5-layer 2D CNN, which yielded much better performance in comparison to the 4-layer network, achieving up to 96% accuracy. To gauge the performance of our approaches, we compared it to Google’s Inception Deep Learning model, which has been pre-trained for hundreds of classes. To suit our purposes, we only re-trained its final layer for the aforementioned classes. We were surprised to find that Inception could achieve up to 70% accuracy when testing for the three classes. Other recent works by various industrial and academic researchers, using the same dataset (LIDC-IDRI), have employed the use of complex 3D CNNs and achieved very high accuracy. We strongly believe that our simplistic approach is very promising.

## Background Information ##
A CT scan scans for lung nodules (a small mass present in the lungs). There are several types:
1. Benign - Harmless nodules that occur as a result of calcification, a hamartoma, or papilloma.
![picture alt](https://img.medscapestatic.com/pi/meds/ckb/44/17144tn.jpg "Title is optional")
2. Malignant - Nodules that exhibit cancer-like growth, and will continue growing larger, having the potential to spread cancer around the body
3. Metastatic - Nodules that are a sign of cancer that has spread from other parts of the body



