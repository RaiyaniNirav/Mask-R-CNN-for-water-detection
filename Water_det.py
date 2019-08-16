# -*- coding: utf-8 -*-
"""
Python code for Water detection using Mask R-CNN first proposed by Girshick et al
and implimented by Matterport in Tensorflow and Keras.

Created on Tue Jul 30 15:49:57 2019

@author: Nirav Raiyani
"""

# Importing the required Python packages
import os
import sys
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


# Get the path of the main file of the project
ROOT_DIR = os.getcwd()

# Getting all the important libraries written for Mask R-CNN (Saved in folder 'mrcnn')

sys.path.append(ROOT_DIR)  # Specifies the path for looking the following packages
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log
import balloon

# Creating the deractory to save logs and weights of the model
MODEL_DIR = os.path.join(ROOT_DIR,"logs")

# Loading the configuration:Object name, No. of epochs and all hyperparameters
config = balloon.BalloonConfig() # Configurations are defined in 'balloon.py' and 'config.py'


# To modify (if needed) some setting in config.
class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# Updating the change made in InferenceConfig
config = InferenceConfig()    
config.display()


# Specifying the devise on which Neural Network is to be loaded
DEVICE = "/cpu:0"


# specifying the mode of operation : Inferance or Training
TEST_MODE = "inference"

# Specifying the basic structure for displying the image on matplotlib
#i.e. Array representing the size of the image.
def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Creating a model in inferance mode
# tf.device specifies the device to use for operation which in current case is
# our model

with tf.device(DEVICE):
# Callng a MaskRCNN model in 'inferance' mode with aboce specified configurations. 
    model = modellib.MaskRCNN(mode= 'inference', model_dir = MODEL_DIR, config = config)

# Declaring the number of classes available in the model for detection.
class_names = ['BG', 'Water']    


    
### Loading the weights of the Model  ###
   
# Specifing the path of the weights
weights_path = "Weight_logs/mask_rcnn_water_0067_1.h5"

#  Loading the weights
print("Loading the weights of the Mask R-CNN", weights_path)
model.load_weights(weights_path, by_name = True)

# Testing the images with trained image

image = cv2.imread('Dataset/Images/image39.jpg')

# Running the image through Mask R-CNN
results = model.detect([image], verbose = 1) 


# Displying the results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], ax = ax, title = "Water detection results" )   
    
    
    
    
'''
# To get the process flow diagram of the whole process
    
model.keras_model.compile(loss='mean_squared_error', optimizer='sgd')  
from keras.utils import plot_model   
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
plot_model(model.keras_model, to_file='HappyModel.png')
SVG(model_to_dot(model.keras_model).create(prog='dot', format='svg'))    
''' 
    
    
    
    
    
    
    
    
    
    