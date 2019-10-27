import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import scipy.misc
import argparse
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--training_data_num', type=int, default=5000)
parser.add_argument('--testing_data_num', type=int, default=1000)
parser.add_argument('--data_set', type=str, default="data/")
args = parser.parse_args()


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


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def create_data_directory():
    # making a checkpoint directory if it doesn't exists
    if not os.path.exists(args.data_set):
        os.mkdir(args.data_set)
    dir_name = args.data_set+"train"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    dir_name = args.data_set+"test"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def generate_samples(dir, csv_name, num):
    # opening a new csv file
    with open(dir + csv_name, 'w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',',
                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # setting the top row of the csv
        file_writer.writerow(['ImageName', 'Row', 'Column', 'Radius'])
        # adding samples
        for i in range(num):
            params, img = noisy_circle(200, 50, 2)
            img_name = 'img' + str(i)+'.png'
            scipy.misc.imsave(dir + img_name, img)
            file_writer.writerow([img_name, params[0], params[1], params[2]])


if __name__ == "__main__":
    # creating directory to store training and test data
    create_data_directory()

    # generating training samples
    generate_samples(args.data_set+"train/", "training.csv", args.training_data_num)
    # generating testing samples
    generate_samples(args.data_set+"test/", "testing.csv", args.testing_data_num)
