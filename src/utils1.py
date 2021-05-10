import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def a_layer(x,units):
    W = weight_variable([x.get_shape().as_list()[1],units])
    b = bias_variable([units])
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W))
    return tf.nn.relu(tf.matmul(x, W) + b)


def bi_layer(x0,x1,sym,dim_pred):
    if sym == False:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        W1p = weight_variable([x1.get_shape().as_list()[1],dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1p))
        return tf.matmul(tf.matmul(x0, W0p), 
                            tf.matmul(x1, W1p),transpose_b=True)
    else:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        return tf.matmul(tf.matmul(x0, W0p),
                            tf.matmul(x1, W0p),transpose_b=True)
# 加上注意力机制
def att_layer(self, seq, my_activation, in_drop=0.0, coef_drop=0.0):
    with tf.name_scope('my_attention_layer'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fst = a_layer(seq, self.dim_pass)
        Wh = weight_variable([self.dim_pass + self.dim_drug, self.dim_drug])
        bh = weight_variable([self.dim_drug])
        v1 = tf.matmul(seq, Wh) + bh  # 自动编码 embedding#self_attention
        v2 = tf.matmul(seq, Wh) + bh
        logits = v1 + tf.transpose(v2, [1, 0])
        coefs = tf.nn.softmax((tf.nn.leaky_relu(logits)))
        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fst)
        ret = tf.contrib.layers.bias_add(vals)
        return my_activation(vals)