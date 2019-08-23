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


 
