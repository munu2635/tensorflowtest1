import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

global_step = tf.Variable(0, trainable = False, name='global_step') # 모델을 저장 할때 쓰는 변수

# CNN 모델을 사용하기위해 2차원 평면과 특성치의 형태를 갖는 구조 
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # 28x28 and 특성치 
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 각각의 변수(가중치)와 레이어의 구성 
# W1 [ 3 3 1 32 ] -> [ 3 3 ] : 커널 크기, 1: 입력값 x의 특성수, 32: 필터 갯수 
# L1 Conv shape = (?, 28, 28, 32)
#    Pool       ->(?, 14, 14, 32)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) # tf.nn.conv2d 이용해 한칸씩 움직이는 컨볼루션 레이어 형성 
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') # 'SAME'은 커널 슬라이딩시 최외곽에사 한칸 밖으로 더 움직이는 옵션 
L1 = tf.nn.relu(L1)

L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# L1 = tf.nn.dropout(L1, keep_prob)

# L2 Conv shape = (?, 14, 14, 64)
#    pool      -> (?, 7, 7, 64)
# W2 의 [3, 3, 32, 64] 에서 32는 L1에서 출력된 W1의 마지막 차원 및 필터의 크기 
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# L2 = tf.nn.dropout(L2, keep_prob)

# FC Layer (Fully Connect Layer) 입력값 7x7x64 -> 출력값 256
# full connect를 위해 직전의 pool 사이즈인 (?, 7, 7, 64)를 참고하여 차원을 줄여준다.
#   Reshape -> (?, 256)
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost, global_step)
# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimizer(cost)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables()) # 앞서 정의한 변수들을 가져옴 // 이 변수들을 저장하거나 학습한 결과를 불러와 담는 변수들로 사용 

checkpoint = tf.train.get_checkpoint_state('./model/cnn')

if checkpoint and tf.train.checkpoint_exists(checkpoint.model_checkpoint_path): # 저장된 기존 모델이 있다면 
    saver.restore(sess, checkpoint.model_checkpoint_path)
else: 
    sess.run(tf.global_variables_initializer()) # 저장된 기존모델이 없다면 

batch_size = 100 
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(1):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})

        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))


saver.save(sess, './model/cnn/cnn.ckpt', global_step=global_step)



is_correct = tf.equal(tf.argmax(model, 1),  tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('accuracy: ', sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, keep_prob: 1 }))

labels = sess.run(model, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, keep_prob:1})

fig = plt.figure()
for i in range(10):
    subplot = fig.add_subplot(2, 5, i + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap=plt.cm.gray_r)

plt.show()
