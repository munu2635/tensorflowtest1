import tensorflow as tf

# if you do calculation, it has input data
X = tf.placeholder(tf.float32, [None, 3])
print(X)

x_data = [[1, 2, 3], [4, 5, 6]]


# Variable is mathenatical optimization variable
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))

expr = tf.matmul(X, W) + b

sess = tf.Session()

sess.run(tf.global_variables_initializer())


print(x_data)
print("-----------w--")
print(sess.run(W))
print("-----------n--")
print(sess.run(b))
print("-----------exr-")

print(sess.run(expr, feed_dict={X:x_data}))

sess.close()

