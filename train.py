import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.resnet import ResNet
from circle_dataset import CirclesDataset
import argparse
import math
import logging
import numpy as np
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, default=14)
parser.add_argument('--data_set', type=str, default='data/')
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=45)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

num_epochs = int(args.num_epochs)
lr = float(args.lr)
start_epoch = 1
batch_size = int(args.batch_size)

# using GPU if available
is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")
min_loss = float("inf")

# Create and configure logger
logging.basicConfig(filename="training_output.txt", format='%(asctime)s %(message)s', filemode='w')
# Creating an object
logger = logging.getLogger()
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


# initializing the parameters in the network
def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


# for reducing the learning after certain epoch
def lr_schedule(learn_rate, epoch):
    optim_factor = 0
    if epoch > 35:
        optim_factor = 3
    if epoch > 25:
        optim_factor = 2
    elif epoch > 15:
        optim_factor = 1
    return learn_rate / math.pow(10, optim_factor)


# Data pre-process
transform_train = transforms.Compose([
    transforms.ToTensor()
])
transform_test = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = CirclesDataset(root_dir='./data/train/', csv_file='training.csv', transform=transform_train)
test_dataset = CirclesDataset(root_dir='./data/test/', csv_file='testing.csv', transform=transform_test)

# creating the data-loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

# # initializing our network
net = ResNet(args.depth, in_channels=1, output=3)

net.apply(conv_init)
print(net)
if is_use_cuda:
    net.to(device)
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

# change loss criterion
criterion = nn.L1Loss()
print("Number of parameters in the network: {}".format(sum(p.numel() for p in net.parameters())))


def train(epoch):
    net.train()
    train_loss = 0
    optimizer = optim.Adam(net.parameters(), lr=lr_schedule(lr, epoch))

    print('Training Epoch: #%d, LR: %.5f' % (epoch, lr_schedule(lr, epoch)))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Training Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f' %
                         (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                          epoch, num_epochs, idx, len(train_dataset) // batch_size,
                          train_loss / (batch_size * (idx + 1))))
        sys.stdout.flush()
    logger.info('Training Epoch [%d/%d] \t\tLoss: %.4f' % (epoch, num_epochs, train_loss / (batch_size * (idx + 1))))


def test(epoch):
    global min_loss
    net.eval()
    test_loss = 0
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        total += labels.size(0)
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f' %
                         (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                          epoch, num_epochs, idx, len(test_dataset) // test_loader.batch_size,
                          test_loss / (batch_size * (idx + 1))))
        sys.stdout.flush()
    logger.info('Testing Epoch [%d/%d] \t\tLoss: %.4f' % (epoch, num_epochs, test_loss / (batch_size * (idx + 1))))

    if min_loss > test_loss/total:
        min_loss = test_loss/total
        if args.save_model:
            file_path = dir_name + "/model" + ".pth"
            torch.save(net.state_dict(), file_path)
            logger.info("Saving the Model")


# making a checkpoint directory if it doesn't exists
dir_name = "checkpoints"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


for _epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()
    train(_epoch)
    print()
    test(_epoch)
    print()
    print()
    end_time = time.time()
    print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))

print('Min Loss@1: %.4f' % min_loss)
logger.info('Min Loss@1: %.4f' % min_loss)
