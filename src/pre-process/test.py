# -*- coding: utf-8 -*-
import os, sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + '/..')
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import KFold
from src.utils import *
from tflearn.activations import relu
from sklearn.cross_validation import train_test_split, StratifiedKFold
import sys
from optparse import OptionParser


a = np.loadtxt('../dataset/positive_results.txt')
print(a.shape)
b = np.loadtxt('../dataset/negative_results.txt')
print(b.shape)
