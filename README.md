# Realtime Age and Gender Detection with-opencv

This repository contains code for a real-time age and gender detection model using OpenCV and the Haar cascades model. The model is capable of detecting the age and gender of people in real-time video feeds or static images.

## Dataset
The dataset used for training this model is available on Kaggle [at](https://www.kaggle.com/datasets/shanmukh05/agedetection). However, it's worth noting that this dataset has very few images of people with dark skin tones, which may lead to bias in the model's accuracy.

## Training

The model was trained using a Kaggle notebook available [here](https://www.kaggle.com/code/rashidrk/age-and-gender-detection). Due to RAM constraints, the training process was limited to only 110 images per age. While this approach can work, it's worth noting that using a small number of images for training may lead to overfitting.

## Face Detection

The code uses the Haar cascades model to detect faces in the input images or video feeds. While this model can work well in certain cases, it's worth noting that there are newer and more advanced face detection models available, such as the Single Shot Multibox Detector (SSD) and the RetinaFace detector. These models may be able to improve the accuracy of your face detection, especially in more challenging images.

## Installation

To use this code, you will need to install the following dependencies:

* Python 3.x
* OpenCV
* numpy
* tensorflow
* argparse

You can install these dependencies using pip. For example:
python
pip install opencv-python
pip install numpy
pip install argparse
pip install tensorflow

## Usage

To run the model, use the following command:
`python opencv face detection.py`
This will open the webcam if available and then the model will start detecing and predicting the age of the faces available
The output will show the detected faces with their predicted age and gender.

## Conclusion

Overall, this repository provides a simple implementation of a real-time age and gender detection model using OpenCV and the Haar cascades model. However, there are several ways to improve the model's accuracy, such as using a more diverse dataset, using more images for training, and using more advanced face detection models. Nonetheless, this code provides a good starting point for anyone interested in exploring age and gender detection using computer vision techniques.
