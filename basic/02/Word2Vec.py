import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sentences = ["i like cat", "i like dog", "i like animal", "dog cat animal",
	"friend cat dog like", "cat fish milk like", "dog fish hate milk like",
	"dog cat eyes like", "i friend like", "friend i hate", "friend i movie book music like",
	"i game music like", "cat dog hate", "dog cat like"]


word_sequence = " ".join(sentences).split()
word_list = " " .join(sentences).split()
word_list = list(set(word_list))

word_dict = {w: i for i, w in enumerate(word_list)}
skip_grams =[]

for i in range(1, len(word_sequence) - 1):
    # (context, target) : ([target index - 1, target index + 1], target)
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i-1]], word_dict[word_sequence[i + 1]]]

    # (target, context[0]), (target, context[1])...
    for w in context:
	    skip_grams.append([target, w])

def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])
        random_labels.append([data[i][1]])
        
    return random_inputs, random_labels


## setting init option
training_epoch = 300

learning_rate = 0.1

batch_size = 20

# word vector's embedding size
embedding_size = 2

num_sampled = 10 # < batch_size

voc_size = len(word_list)

## model
inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))

select_embed = tf.nn.embedding_lookup(embeddings, inputs)

nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, labels,
	select_embed, num_sampled, voc_size))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(1, training_epoch+1):
	    batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

	    _, loss_val = sess.run([train_op, loss],feed_dict={inputs: batch_inputs, labels: batch_labels})

	    if step % 10 == 0:
		    print("loss at step ", step, ": ", loss_val)

    trained_embeddings = embeddings.eval()

    for i, label in enumerate(word_list):
	    x, y = trained_embeddings[i]
	    plt.scatter(x,y)
	    plt.annotate(label, xy=(x, y), xytext=(5, 2),
			testcoords='offset points', ha='right', va='bottom')
plt.show()
