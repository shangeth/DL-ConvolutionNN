import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
from cnn import CNN


np.random.seed(1)


def main():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    X_train_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    Y_train_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    X_test_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    Y_test_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    Y_train_orig = Y_train_orig.reshape((1, Y_train_orig.shape[0]))
    Y_test_orig = Y_test_orig.reshape((1, Y_test_orig.shape[0]))




    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    conv_layers = {}





    clf = CNN()
    clf.fit(X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    main()