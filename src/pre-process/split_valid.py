# -*- coding: utf-8 -*-

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
import os

parser = OptionParser()
parser.add_option("-d", "--d", default=1024, help="The embedding dimension d / embeddig 维数d")
parser.add_option("-n", "--n", default=1, help="global norm to be clipped")
parser.add_option("-k", "--k", default=512, help="The dimension of project matrices k / 项目矩阵的维数")
parser.add_option("-t", "--t", default="o", help="Test scenario / 测试场景")
parser.add_option("-r", "--r", default="ten", help="positive negative ratio. "
                                                   "Two choices: ten and all, the former one sets the positive:negative = 1:10, the latter one considers all unknown DTIs as negative examples.")

(opts, args) = parser.parse_args()

for r in range(1):
    print('sample round', r + 1)

    dti_o = np.load('../dataset/' + opts.r + '_valid.npy')
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_o)[0]):
        j1 = dti_o[i][0]
        j2 = dti_o[i][1]
        # print(j1, j2)
        if int(dti_o[i][2]) == 1:
            whole_positive_index.append([j1, j2])
        elif int(dti_o[i][2]) == 0:
            whole_negative_index.append([j1, j2])

    data_set = np.zeros((len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    print('positive: ', count)
    np.save('../dataset/' + opts.r + '_valid_positive.npy', data_set)
    data_set = np.zeros((len(whole_negative_index), 3), dtype=int)
    count = 0
    for i in whole_negative_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 0
        count += 1
    print('negative: ', count)
    np.save('../dataset/' + opts.r + '_valid_negative.npy', data_set)

    a = np.load('../dataset/' + opts.r + '_valid_positive.npy')
    print(a.shape)
