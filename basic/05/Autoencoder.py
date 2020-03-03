# 대표적인 비지도(Unsupervised) 학습
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

global_step = tf.Variable(0, trainable = False, name='global_step') # 모델을 저장 할때 쓰는 변수
## option setting
learning_rate = 0.01 
training_epoch = 100
batch_size = 100 
n_hidden = 256
n_input = 28*28

# 신경망 모델 구성 비지도 학습이므로 Y 데이터가 없음 
X = tf.compat.v1.placeholder(tf.float32, [None, n_input])

# encoder 와  decoder의 가중치와 편향 변수를 설정한다. 
# input -> encode -> decode -> output
W_encode = tf.Variable(tf.random.normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random.normal([n_hidden]))

# encoder = sigmoid(X * W + b)
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))


# encode 의 아웃풋 크기를 입력값보다 작은 크기로 만들어 정보를 압축 - 특성을 뽑아내고 
# decode 의 출력을 입력 값과 동일한 크기를 갖도록 하여 입력과 똑같은 아웃풓을 만들어 내도록 한다. 

W_decode = tf.Variable(tf.random.normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random.normal([n_input]))

# decoder = sigmoid(encoder*W + b) 디코더가 최종모델 
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

# 입력값과 같은 데이터를 갖고있는지 확인 
cost = tf.reduce_mean(tf.pow(X - decoder, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost, global_step)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables()) # 앞서 정의한 변수들을 가져옴 // 이 변수들을 저장하거나 학습한 결과를 불러와 담는 변수들로 사용 

checkpoint = tf.train.get_checkpoint_state('./model/autoencoder')

if checkpoint and tf.train.checkpoint_exists(checkpoint.model_checkpoint_path): # 저장된 기존 모델이 있다면 
    saver.restore(sess, checkpoint.model_checkpoint_path)
else: 
    sess.run(tf.global_variables_initializer()) # 저장된 기존모델이 없다면 

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(training_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        total_cost += cost_val 

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.4f}'.format(total_cost/total_batch))


saver.save(sess, './model/autoencoder/autoencoder.ckpt', global_step=global_step)

sample_size = 10

samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

plt.show()
