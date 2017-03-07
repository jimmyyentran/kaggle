import tensorflow as tf
import sys
from IPython import embed

def save():
    with tf.name_scope('input'):
        v1 = tf.placeholder(tf.float32, name="v1") 
        v2 = tf.placeholder(tf.float32, name="v2")
    v3 = tf.mul(v1, v2)
    vx = tf.Variable(10.0, name="vx")
    v4 = tf.add(v3, vx, name="v4")
    saver = tf.train.Saver([vx])
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    sess.run(vx.assign(tf.add(vx, vx)))
    result = sess.run(v4, feed_dict={v1:12.0, v2:3.3})
    print(result)
    saver.save(sess, "./model_ex1")

def restore():
    saver = tf.train.import_meta_graph("./model_ex1.meta")
    sess = tf.Session()
    saver.restore(sess, "./model_ex1")
    #  result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 3.3})
    #  print(result)
    embed()

if sys.argv[1].lower() == 'save':
    save()
else:
    restore()
