import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tflearn.initializations import truncated_normal


from tflearn.activations import relu

def focal_loss(y_pred, y_true, alpha=0.25, gamma=2., epsilon=1e-6):
    positive = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    negative = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return tf.reduce_sum(-alpha * tf.pow(1. - positive, gamma) * tf.log(positive + epsilon) - \
                         (1 - alpha) * tf.pow(negative, gamma) * tf.log(1. - negative + epsilon))


def focal_loss2(pred, y, alpha=0.1, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     pred: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     y: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    #     zeros = tf.zeros_like(pred, dtype=pred.dtype)
    #
    # # For positive prediction, only need consider front part loss, back part is 0;
    # # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    # pos_p_sub = tf.where(y > zeros, y - pred, zeros)  # positive sample 寻找正样本，并进行填充
    #
    # # For negative prediction, only need consider back part loss, front part is 0;
    # # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    # neg_p_sub = tf.where(y > zeros, zeros, pred)  # negative sample 寻找负样本，并进行填充
    # per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
    #                       - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))
    per_entry_cross_ent = tf.multiply(y, pred)
    return tf.reduce_sum(per_entry_cross_ent)


def AconC(x):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """
    print(x.shape)
    width = x.shape
    print('width ', width)
    shape = [1, width[1]]
    p1 = tf.Variable(tf.random.normal(shape))
    p2 = tf.Variable(tf.random.normal(shape))
    beta = tf.Variable(tf.ones(shape))

    sigmoid = tf.sigmoid

    return (p1 * x - p2 * x) * sigmoid(beta * (p1 * x - p2 * x)) + p2 * x


def MetaAconC(x, r=16):
    width = x.shape[1]
    print('width ', width)
    shape = [1, width]
    p1 = tf.Variable(tf.random.normal(shape))
    p2 = tf.Variable(tf.random.normal(shape))

    # fc1 = tf.layers.conv1d(x, max(r, width // r), kernel_size=1, strides=1)
    # bn1 = tf.layers.batch_normalization(max(r, width // r))
    #
    # bn2 = tf.layers.batch_normalization(width)

    sigmoid = tf.sigmoid
    # fc2 = tf.layers.conv2d(bn1(fc1), strides=1, bias=True)
    beta = AconC(x)

    return (p1 * x - p2 * x) * sigmoid(beta * (p1 * x - p2 * x)) + p2 * x


# relu = AconC


def weight_variable(shape, name=None):
    # 产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name)


def bias_variable(shape, name=None):
    # 初始化一个维度为shape，数值为0.1的张量
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name)


def a_layer(x, units):
    W = weight_variable([x.get_shape().as_list()[1], units])
    b = bias_variable([units])
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W))
    return relu(tf.matmul(x, W) + b)


def bi_layer(x0, x1, sym, dim_pred, name=None):
    if sym == False:
        W0p = weight_variable([x0.get_shape().as_list()[1], dim_pred])
        W1p = weight_variable([x1.get_shape().as_list()[1], dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1p))
        return tf.matmul(tf.matmul(x0, W0p),
                         tf.matmul(x1, W1p), transpose_b=True, name=name)
    else:
        W0p = weight_variable([x0.get_shape().as_list()[1], dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        return tf.matmul(tf.matmul(x0, W0p),
                         tf.matmul(x1, W0p), transpose_b=True, name=name)
