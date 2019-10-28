import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
from models.resnet import ResNet
import torch
import torch.nn as nn
from PIL import Image
import scipy.misc
import os
import torchvision.transforms as transforms


loader = transforms.Compose([transforms.ToTensor()])

# checking if the GPU is available for inference
is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")

# initializing the model
net = ResNet(depth=14, in_channels=1, output=3)
# moving the net to GPU for testing
if is_use_cuda:
    net.to(device)
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

# loading the network parameters
net.load_state_dict(torch.load("./checkpoints/model.pth"))
net.eval()


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)
    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle():
    image = Image.open("./img.png")
    image = loader(image).float().resize(1, 1, 200, 200)
    output = net(image.cuda()).cpu().detach().numpy()
    # Fill in this function
    return output[0][0], output[0][1], output[0][2]


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


if __name__ == "__main__":
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        scipy.misc.imsave("img.png", img)
        detected = find_circle()
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())
    os.remove("img.png")
