# Cats and Dogs Classification CNN


![](https://raw.githubusercontent.com/twhipple/Cats_and_Dogs_Classification_CNN/main/Images/alec-favale-Ivzo69e18nk-unsplash.jpg)

*Training a model to recognize cats and dogs. Source: Alec Favale, unsplash.com*


## Intro
My first CNN using color images. The idea is to build a model that can recognize the difference between cat and dog images.



## README Outline
* Introduction and README outline
* Repo Contents
* Libraries & Prerequisites
* Results
* Future Work
* Built With, Contributors, Authors, Acknowledgments


![](https://raw.githubusercontent.com/twhipple/Cats_and_Dogs_Classification_CNN/main/Images/SequentialModelSummary.png)

*Sample of handwritten digits from the MNIST dataset.*


## Repo Contents
This repo contains the following:
* README.md - this is where you are now!
* Notebook.ipynb - the Jupyter Notebook containing the finalized code for this project.
* LICENSE.md - the required license information.
* dataset - the dataset can be found in the scikit-learn datasets and is not a separate file.
* CONTRIBUTING.md
* Images


## Libraries & Prerequisites
These are the libraries that I used in this project.

* import numpy as np
* import pandas as pd
* import random as rn
* Tensorflow
* import tensorflow.random as tfr
* import tensorflow.keras as keras
* from tensorflow.keras.models import Sequential, load_model
* from tensorflow.keras.layers import Dense, Dropout, Flatten
* from tensorflow.keras.layers import Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization
* from tensorflow.keras import backend as K
* from tensorflow.keras.utils import to_categorical
* from tensorflow.keras.optimizers import RMSprop, Adam
* from tensorflow.keras.preprocessing.image import ImageDataGenerator
* from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
* Chart
* import matplotlib.pyplot as plt
* import matplotlib.image as mpimg
* import seaborn as sns
* from sklearn.metrics import classification_report
* from PIL import Image
* import os
* import cv2


## Conclusions
So the Convolutional Nueral Network did slightly better than the basic Multi-layer Perceptron NN. Both had pretty high valadation accuracy results even after only a couple Epochs.


## Future Work
With some help from my former instructor, Jeff Herman, I was finally able to get these simple models to run. I'm sure there are more parameters I could adjust, or add more layers to the perceptron in either model. It didn't seem like running more Epochs would make a difference, though I could change the batch size. I would also really like to see how this model works with additonal hand-written digits.


![](https://raw.githubusercontent.com/twhipple/Cats_and_Dogs_Classification_CNN/main/Images/TrainvsEpochs.png)

*Training Accuracy graph.*


## Built With:
Jupyter Notebook
Python 3.0
scikit.learn

## Contributing
Please read CONTRIBUTING.md for details

## Authors
Thomas Whipple

## License
Please read LICENSE.md for details

## Acknowledgments
Thank you to Jeff Herman for all of your support and help with code!
Thank you to scikit-learn for the dataset.
