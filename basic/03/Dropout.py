# 과적합 방지 기법 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


global_step = tf.Variable(0, trainable=False, name='global_step')
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# what is stddev 
# 1 layer 
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
# tensorflow 내장 함수를 이용하여 dropout을 적용 - 적용 layer와 확률만 넣어주면 된다
L1 = tf.nn.dropout(L1, keep_prob)

# 2 layer 
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)

# 3 layer
W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost, global_step)

# model 

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

checkpoint = tf.train.get_checkpoint_state('./model/dropout')

if checkpoint and tf.train.checkpoint_exists(checkpoint.model_checkpoint_path):
    saver.restore(sess, checkpoint.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(10):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # tensorflow의 mnist next_batch 함수를 이용해 지정한 크기만큼 학습데이터를 갖고온다.

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
        total_cost += cost_val

    print('Epoch:', '%04d' % ( epoch + 1 ), 'Avg. cost=', '{:.3f}'.format(total_cost / total_batch))


saver.save(sess, './model/dropout/mnist.ckpt', global_step=global_step)

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

labels = sess.run(model, feed_dict={X: mnist.test.images,
                                    Y: mnist.test.labels,
                                    keep_prob:1})

fig = plt.figure()
for i in range(10):
    subplot = fig.add_subplot(2, 5, i + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap=plt.cm.gray_r)

plt.show()





