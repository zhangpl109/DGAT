# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:02:48 2018

@author: fangp

Cross validation for NeoDTI.

modified by Kui Yao on 2021/01/21
"""
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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 创建saver
saver = tf.train.Saver()
# tf.device('/gpu:0')
parser = OptionParser()
parser.add_option("-d", "--d", default=1024, help="The embedding dimension d / embeddig 维数d")
parser.add_option("-n", "--n", default=1, help="global norm to be clipped")
parser.add_option("-k", "--k", default=512, help="The dimension of project matrices k / 项目矩阵的维数")
parser.add_option("-t", "--t", default="o", help="Test scenario / 测试场景")
parser.add_option("-r", "--r", default="ten", help="positive negative ratio. "
                                                   "Two choices: ten and all, the former one sets the positive:negative = 1:10, the latter one considers all unknown DTIs as negative examples.")

(opts, args) = parser.parse_args()


# load network
network_path = '../data/'
if opts.t == 'o':
    # 药物/蛋白质
    dti_o = np.loadtxt(network_path + 'mat_drug_protein.txt')
else:
    dti_o = np.loadtxt(network_path + 'mat_drug_protein_' + opts.t + '.txt')

whole_positive_index = []
whole_negative_index = []
for i in range(np.shape(dti_o)[0]):
    for j in range(np.shape(dti_o)[1]):
        if int(dti_o[i][j]) == 1:
            whole_positive_index.append([i, j])
        elif int(dti_o[i][j]) == 0:
            whole_negative_index.append([i, j])

if opts.r == 'ten':
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=10 * len(whole_positive_index), replace=False)
elif opts.r == 'all':
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)), size=len(whole_negative_index),
                                             replace=False)
else:
    print('wrong positive negative ratio')
data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
count = 0
for i in whole_positive_index:
    data_set[count][0] = i[0]
    data_set[count][1] = i[1]
    data_set[count][2] = 1
    count += 1
for i in negative_sample_index:
    data_set[count][0] = whole_negative_index[i][0]
    data_set[count][1] = whole_negative_index[i][1]
    data_set[count][2] = 0
    count += 1

DTItrain, DTIvalid = train_test_split(data_set, test_size=0.4, random_state=2021)
DTItest, DTIvalid = train_test_split(DTIvalid, test_size=0.5, random_state=2021)

print(len(DTItrain),len(DTItest),len(DTIvalid))

# np.save('dataset/'+opts.r+'_train.npy',DTItrain)
# np.save('dataset/'+opts.r+'_valid.npy',DTIvalid)
# np.save('dataset/'+opts.r+'_test.npy',DTItest)

print(len(np.load('dataset/'+opts.r+'_train.npy')))
