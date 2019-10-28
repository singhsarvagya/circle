# Circle
The purpose of this exercise is to detection dimensions of a circle drawn in a 200X200 pixel grayscale image using deep neural network

## Network 

For the regression tasks I am using a Resnet14. Resnet is a state-of-the-art deep learning architecture used for object detections. The ResNet used here is adopted from https://github.com/bearpaw/pytorch-classification, and is modified for this regression task. The parameter has totol 14 layers: 
  * Initial Convolution layer
  * 6 ResNet block with 2 convolution layers each
  * one fully connected layer
  
This is a little more than maximum numbers of layers aloowed i.e. 10, but the model contains only 172k parameters which is much less than the maximum amount of params allowed i.e. 1M. 

## Loss Function

I have used L1 loss function instead of L2(MSELoss) for this task. Although MSELoss is preferred for regression tasks over L1 loss, but for this exercise L1 performed much better. 

## Optimizer 

I have used Adam optimizer for this task as it has shown to optimize better than SGD in scenrios where loss function is not strictly convex. 

## Usage

First you need to create a dataset 
```
python3 generate_data.py --training_data_num <# of training samples> --testing_data_num <# of testing exmample>
```
For training simply run. You can pick options such as, model depth, dataset loc, saving model, epochs, leanring rate and batch size.
```
python3 train.py
```
To test the trained model, simly run: 
```
python3 main.py
```

