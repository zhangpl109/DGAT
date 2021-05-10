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

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 创建saver
saver = tf.train.Saver()
parser = OptionParser()
parser.add_option("-d", "--d", default=1024, help="The embedding dimension d / embeddig 维数d")
parser.add_option("-n", "--n", default=1, help="global norm to be clipped")
parser.add_option("-k", "--k", default=512, help="The dimension of project matrices k / 项目矩阵的维数")
parser.add_option("-t", "--t", default="o", help="Test scenario / 测试场景")
parser.add_option("-r", "--r", default="ten", help="positive negative ratio. "
                                                   "Two choices: ten and all, the former one sets the positive:negative = 1:10, the latter one considers all unknown DTIs as negative examples.")

(opts, args) = parser.parse_args()


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        # 将矩阵a_matrix对角元素设置为0
        np.fill_diagonal(a_matrix, 0)
    a_matrix = a_matrix.astype(float)
    # 对每行数据求和
    row_sums = a_matrix.sum(axis=1) + 1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix


# load network
network_path = '../../data/'

# 药物/药物
drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
# print 'loaded drug drug', check_symmetric(drug_drug), np.shape(drug_drug)
true_drug = 708  # First [0:708] are drugs, the rest are compounds retrieved from ZINC15 database
# 基于药物化学结构的药物和化合物相似度评分([0,708)为药物，其余为化合物
drug_chemical = np.loadtxt(network_path + 'Similarity_Matrix_Drugs.txt')
drug_chemical = drug_chemical[:true_drug, :true_drug]
# print 'loaded drug chemical', check_symmetric(drug_chemical), np.shape(drug_chemical)
# 药物/疾病
drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
# print 'loaded drug disease', np.shape(drug_disease)
# 药物/副作用
drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')
# print 'loaded drug sideffect', np.shape(drug_sideeffect)
# 疾病/药物
disease_drug = drug_disease.T
# 副作用/药物
sideeffect_drug = drug_sideeffect.T

# 蛋白质/蛋白质
protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
# print 'loaded protein protein', check_symmetric(protein_protein), np.shape(protein_protein)
# 基于蛋白质一级序列的蛋白质相似性评分
protein_sequence = np.loadtxt(network_path + 'Similarity_Matrix_Proteins.txt')
# print 'loaded protein sequence', check_symmetric(protein_sequence), np.shape(protein_sequence)
# 蛋白质/疾病
protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')
# print 'loaded protein disease', np.shape(protein_disease)
# 疾病/蛋白质
disease_protein = protein_disease.T

# normalize network for mean pooling aggregation
drug_drug_normalize = row_normalize(drug_drug, True)
drug_chemical_normalize = row_normalize(drug_chemical, True)
drug_disease_normalize = row_normalize(drug_disease, False)
drug_sideeffect_normalize = row_normalize(drug_sideeffect, False)

protein_protein_normalize = row_normalize(protein_protein, True)
protein_sequence_normalize = row_normalize(protein_sequence, True)
protein_disease_normalize = row_normalize(protein_disease, False)

disease_drug_normalize = row_normalize(disease_drug, False)
disease_protein_normalize = row_normalize(disease_protein, False)
sideeffect_drug_normalize = row_normalize(sideeffect_drug, False)

# define computation graph
num_drug = len(drug_drug_normalize)
num_protein = len(protein_protein_normalize)
num_disease = len(disease_protein_normalize)
num_sideeffect = len(sideeffect_drug_normalize)

dim_drug = int(opts.d)
dim_protein = int(opts.d)
dim_disease = int(opts.d)
dim_sideeffect = int(opts.d)
dim_pred = int(opts.k)
dim_pass = int(opts.d)


def train_and_evaluate(DTItrain, DTIvalid, DTItest, graph, verbose=True, num_steps=4000, count0=0, count1=0,
                       negative_results=None,
                       positive_results=None):
    drug_protein = np.zeros((num_drug, num_protein))
    mask = np.zeros((num_drug, num_protein))
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1
    protein_drug = drug_protein.T

    drug_protein_normalize = row_normalize(drug_protein, False)
    protein_drug_normalize = row_normalize(protein_drug, False)

    feed_dict = {
        a_drug_protein: drug_protein,
        a_drug_protein_normalize: drug_protein_normalize,
        a_protein_drug: protein_drug,
        a_protein_drug_normalize: protein_drug_normalize,
        a_drug_protein_mask: mask,
        a_learning_rate: lr
    }

    # results = sess.run(a_drug_protein_reconstruct, feed_dict)
    # print(results)
    results_drug_vector1, results_protein_vector1 = sess.run([a_drug_vector1, a_protein_vector1], feed_dict)

    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0

    for i in range(num_steps):
        print('step', i, 'total and dtiloss')

        pred_list = []
        ground_truth = []
        for ele in DTItrain:
            if ele[2] == 0:
                negative_results += ((results_drug_vector1[ele[0]].T * results_protein_vector1[ele[1]]))
                count0 += 1
            else:
                positive_results += ((results_drug_vector1[ele[0]].T * results_protein_vector1[ele[1]]))
                count1 += 1
    return count0, count1, positive_results, negative_results


classify = 'positive'
test_auc_round = []
test_aupr_round = []
feature_result = np.zeros((num_drug, num_protein))
positive_results = np.zeros((1024, 1024))
negative_results = np.zeros((1024, 1024))
for r in range(1):
    print('sample round', r + 1)

    data_set = np.load('../dataset/' + opts.r + '_valid_' + classify + '.npy')
    train_data_set = np.load('../dataset/' + opts.r + '_valid.npy')
    print(len(data_set))
    lr = 0.001

    sess = tf.Session()
    # 先加载图和参数变量
    saver = tf.train.import_meta_graph('../save/model.ckpt-2999.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../save'))

    # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
    graph = tf.get_default_graph()
    a_drug_protein = graph.get_tensor_by_name("drug_protein:0")
    a_drug_protein_normalize = graph.get_tensor_by_name("drug_protein_normalize:0")
    a_protein_drug = graph.get_tensor_by_name("protein_drug:0")
    a_protein_drug_normalize = graph.get_tensor_by_name("protein_drug_normalize:0")
    a_drug_protein_mask = graph.get_tensor_by_name("drug_protein_mask:0")
    a_learning_rate = graph.get_tensor_by_name("learning_rate:0")

    a_drug_vector1 = graph.get_tensor_by_name("drug_vector1:0")
    a_protein_vector1 = graph.get_tensor_by_name("protein_vector1:0")

    # 接下来，访问你想要执行的op
    a_drug_protein_reconstruct = graph.get_tensor_by_name("drug_protein_reconstruct:0")
    if opts.t == 'unique':
        pass
    else:
        test_auc_fold = []
        test_aupr_fold = []
        rs = np.random.randint(0, 1000, 1)[0]
        kf = StratifiedKFold(train_data_set[:, 2], n_folds=10, shuffle=True, random_state=rs)

        count0 = count1 = 0
        for train_index, test_index in kf:
            DTItrain, DTIvalid = train_data_set[train_index], train_data_set[test_index]
            # for DTItest in data_set:
            DTItest = data_set
            graph = tf.get_default_graph()

            count0, count1, positive_results1, negative_results1 = train_and_evaluate(DTItrain=DTItrain,
                                                                                      DTIvalid=DTIvalid,
                                                                                      DTItest=DTItest,
                                                                                      graph=graph, num_steps=1,
                                                                                      count0=count0, count1=count1,
                                                                                      negative_results=negative_results,
                                                                                      positive_results=positive_results)
            positive_results += positive_results1
            negative_results += negative_results1
            print(count0,count1)

    np.savetxt('../dataset/positive_results.txt', positive_results / count1)
    np.savetxt('../dataset/negative_results.txt', negative_results / count0)
