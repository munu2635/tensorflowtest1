import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True) # one_hot 데이터 

global_step = tf.Variable(0, trainable = False, name='global_step') # 모델을 저장 할때 쓰는 변수
# make model 
X = tf.placeholder(tf.float32, [None, 784]) # 28 x 28 pix == 784 
Y = tf.placeholder(tf.float32, [None, 10]) # 0 ~ 9 

# layers 784 -> 256 (hidden layer 1) -> 256 (hidden layer 2) -> 10 (0 ~ 1 data)
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01)) # 784 x 256의 랜덤 배열 = 가중치 1
L1 = tf.nn.relu(tf.matmul(X, W1)) # relu 함수의 hidden layer 1 -> 1 x 256의 결과 배열 

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01)) # 256 x 256 의 랜덤 배열 = 가중치 2
L2 = tf.nn.relu(tf.matmul(L1, W2)) # relu 함수의 hidden layer 2 -> 1 x 256의 결과 배열 

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01)) # 256 x 10의 랜덤 배열 = 가중치 3
model = tf.matmul(L2, W3) # 1 x 10 의 결과 배열 


# reduce_mean // softmax -- cost function 오차율 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y)) 
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost, global_step=global_step) # cost function을 줄이는 방식 

# model 
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables()) # 앞서 정의한 변수들을 가져옴 // 이 변수들을 저장하거나 학습한 결과를 불러와 담는 변수들로 사용 

ckpt = tf.train.get_checkpoint_state('./model/mnist')

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path): # 저장된 기존 모델이 있다면 
    saver.restore(sess, ckpt.model_checkpoint_path)
else: 
    sess.run(tf.global_variables_initializer()) # 저장된 기존모델이 없다면 


batch_size = 100 # 한번에 학습 할 데이터양
total_batch = int(mnist.train.num_examples / batch_size) # 총 배치 갯수 

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # tensorflow의 mnist next_batch 함수를 이용해 지정한 크기만큼 학습데이터를 갖고온다.

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % ( epoch + 1 ), 'Avg. cost=', '{:.3f}'.format(total_cost / total_batch))


saver.save(sess, './model/mnist/mnist.ckpt', global_step=global_step)

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('predict : ', sess.run(prediction, feed_dict={X: mnist.test.images}))
print('real : ', sess.run(target, feed_dict={Y: mnist.test.labels}))



is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
