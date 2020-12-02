# Cats and Dogs Classification CNN


![](https://raw.githubusercontent.com/twhipple/Cats_and_Dogs_Classification_CNN/main/Images/alec-favale-Ivzo69e18nk-unsplash.jpg)

*Training a model to recognize cats and dogs. Source: Alec Favale, unsplash.com*


## Intro
My first CNN using color images. The idea is to build a model that can recognize the difference between cat and dog images.

10000 .jpg image files - 217.23 MB. Broken up into a Training set (8000 images) and a Testing set (2000 images). Each of these sets are split evenly into a Cats folder and a Dogs folder. 


## README Outline
* Introduction and README outline
* Repo Contents
* Libraries & Prerequisites
* Results
* Future Work
* Built With, Contributors, Authors, Acknowledgments


![](https://raw.githubusercontent.com/twhipple/Cats_and_Dogs_Classification_CNN/main/Images/SequentialModelSummary.png)

*Summary of my Sequential CNN model.*


## Repo Contents
This repo contains the following:
* README.md - this is where you are now!
* Cats_Dogs_Classification_Notebook.ipynb - the Jupyter Notebook containing the finalized code for this project.
* LICENSE.md - the required license information.
* DATA - the dataset can be found on Kaggle: https://www.kaggle.com/chetankv/dogs-cats-images
* CONTRIBUTING.md
* IMAGES


## Libraries & Prerequisites
These are the libraries that I used in this project.

* import numpy as np 
* import matplotlib.pyplot as plt 
* import os
* from PIL import Image
* Keras Libraries
* import keras
* from keras.models import Sequential
* from keras.layers import Conv2D
* from keras.layers import MaxPooling2D
* from keras.layers import Flatten
* from keras.layers import Dense
* from keras.preprocessing.image import ImageDataGenerator, load_img
* from sklearn.metrics import classification_report, confusion_matrix


## Conclusions
I tweeked the model a few different times and finally had an Accuracy: 0.8190 - I even was able to get it to recognize the image from my prediction sample. 


## Future Work
As mentioned in my notebook, I had to physically manipulate the data - I split up the testing group into a Test and a Validation group. I know there is a way to do this using Python code but I couldn't figure it out or get it to work. Therefore, one thing I would like to learn in the future is how to do this. I believe there are also a number of different parameters as well as many more post model result processes that I could adjust and review (such as look at a confusion matrix, or model optimizers).


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
Thank you to kaggle and 'chetanimravan' for the dataset.