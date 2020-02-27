
import tensorflow as tf


a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(c)

sess = tf.Session()

print(sess.run([a,b,c]))

sess.close()
