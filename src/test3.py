import tensorflow as tf

path_name = './save/model.ckpt-1'
saver = tf.train.import_meta_graph(path_name + '.meta')
with tf.Session() as sess:
    saver.restore(sess, path_name)
    for v in tf.get_default_graph().get_collection("variables"):
        print('From variables collection ', v)
        # try:
        print(v.read_value().eval())
        # except:
        #     print("000")
