from src.utils import *
import tensorflow as tf

# y_true = [[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]
# y_pred = [[0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.58],
#           [0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.58]]
# alpha = 0.25
# gamma = 2.
# epsilon = 1e-6
# # c = focal_loss(b, a)
# positive = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
# negative = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
# with tf.Session() as sess:
#     c = sess.run(positive)
#     print(c)
#     c = sess.run(negative)
#     print(c)
#     d = (-alpha * tf.pow(1. - positive, gamma) * tf.log(positive + epsilon) - \
#          (1 - alpha) * tf.pow(negative, gamma) * tf.log(1. - negative + epsilon))
#     print(sess.run(d))
# r1 = [0.9561426657195822, 0.9589592682368779, 0.9584772526263745, 0.9565347652766214, 0.9602830071752093,
#       0.9542178449875474, 0.9568851407985509, 0.9574834046902463, 0.9540472777158063, 0.9542219904469755]
# r2 = [0.8679881372889623, 0.8676463997418125, 0.8731549722772749, 0.8673889342946802, 0.884661542125625,
#       0.8598520388198077, 0.8775569598929008, 0.8696884876915597, 0.8693882881973268, 0.8645523564618334]
#
# A1 = [0.95561868610323, 0.9525956357770717, 0.9590413059527562, 0.95618432520116, 0.952423926188481, 0.9573767440119021,
#       0.9559907754686698, 0.9557986262671578, 0.9567433475620545, 0.9561866561450023]
# A2 = [0.8698027784268424, 0.8678486216347794, 0.8693703108184522, 0.8729234891033064, 0.8719416895473351,
#       0.8813126355686544, 0.8688504427341002, 0.8779024088939517, 0.8698799079131427, 0.8766321170143756]
#
# fun = np.mean
#
# print(fun(r1))
# print(fun(r2))
# print(fun(A1))
# print(fun(A2))

