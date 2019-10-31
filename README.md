# Circle
The purpose of this exercise is to detection dimensions of a circle drawn in a 200X200 pixel grayscale image with added noise using deep neural network

## Network 

For the regression tasks I am using a Resnet14. Resnet is a state-of-the-art deep learning architecture used for object detections. The ResNet used here is adopted from https://github.com/bearpaw/pytorch-classification which is used for CIFAR10 image classification, and I have modified for this regression task. The parameter has total 14 layers: 
  * Initial Convolution layer
  * 6 ResNet block with 2 convolution layers each
  * one fully connected layer
  
This is a little more than maximum numbers of layers aloowed i.e. 10, but the model contains only 700k parameters which is much less than the maximum amount of params allowed i.e. 1M. The extra layers are there due to the general structure of the resnet and very crucial to the perfomance of the model.

## Loss Function

I have used L1 loss function instead of L2(MSELoss) for this task. Although MSELoss is preferred for regression tasks over L1 loss, but for this exercise L1 performed much better than MSE loss.

## Optimizer 

I have used Adam optimizer for this task as it has shown converge the model faster than SGD in scenrios where loss function is not  convex. 

## Requirements

The package required to the run the code are listed in requirements.txt and can be installed using the follwing command
```pip install -r requirements.txt```

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

## Training

The model was trained using 10,000 training examples and 1,000 test examples. All the exmample are generated using the generate_data.py script and have noise level 2. The model have been trained for 45 epochs with batch size of 16. 

### Improvments in training
* The training example could have included more example with different noise level which would have made the model more robust
* Layer level dropout or FC dropout could have been used to make sure that all the parameters are active in the finally trained model
* The learning rate schedular could have been modified to reduce the learning everytime the loss palteaus. One can notice from the log files that the loss hasn't conpletely plateuded when the learning rate is being adjusted.

## Results

The loss values are in the file training_output.txt and the model is stored in checkpoints/model.pth. The results are as follows: 

| Noise Level      | Accuracy    |
| -----------      | ----------- |
| 2                | 97.2%       |
| 1                | 97.6%       |
| 0                | 98.7%       |
