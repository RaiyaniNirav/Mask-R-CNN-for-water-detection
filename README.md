# Mask-R-CNN-for-water-detection
This is an implimentation of Mask R-CNN proposed by Facebook AI Research (Girshick et al) for water detection implemented in python3, Keras and Tensorflow based on (https://arxiv.org/abs/1703.06870). The Mask R-CNN model performes water curtain detection, draws bounding box around it and applies segmentation mask. 

![](assets/Water_Det.gif)

The repository includes:
* The source code of Mask R-CNN with Resnet 101 and FPN(Feature Pyramid Network) as backbone.
* The implimentation detail of Mask R-CNN for water detection.
* A step by step procedure to train Mask R-CNN on your own dataset.
* Pre-trained weights of MS coco dataset.
* A Python code to run object detection on video and webcam.

The implimentation can be devided in to three steps.
1. Preparation of Datasets
2. Mask R-CNN Training
3. Testing the Model

# 1. Preparation of dataset
Mask R-CNN model training requires annotated images. This work uses ```VGG-Image annotator-1.0.6``` for annotation, which can be downloaded from [here](http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html). The VGG annotation tool is an online html web page which does not require any downloading and is very easy to use. The annotation tool gives annotation in ```.json``` file containg the x and y co-ordinates of annotated pixels.

```

CODE TIP:
If you are using VGG-Image annotator-1.0.6, than the code of this repository doesnot require any modification
while training.
Side Note: If your data set is in video instead of images, this repository includes the python code to sample
video at predefined sampling rate into images. 
```
