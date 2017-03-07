import tensorflow as tf
import sys
from IPython import embed

model = str(sys.argv[1])
meta = model + ".meta"

saver = tf.train.import_meta_graph(meta)
for i in tf.get_collection('variables'):
    print i.name

embed()

sess = tf.Session()
saver.restore(sess, model)
result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 3.3})
print(result)
