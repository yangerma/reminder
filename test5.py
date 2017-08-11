import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

PicSize=784

#features: [None, PicSize]
#labels: [None, 10]
#W: [PicSize, 10]
#b: [None, 10]
def model(features, labels, mode):
	W = tf.get_variable('W', [PicSize, 10], dtype=tf.float32)
	b = tf.get_variable('b', [10], dtype=tf.float32)
	y = tf.matmul(features['x'], W) + b
	#print(features['x'].shape)
	
	
	#s = tf.reduce_sum(tf.exp(y))

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y))
	
	global_step = tf.train.get_global_step()
	optimizer = tf.train.GradientDescentOptimizer(0.5)
	train = tf.group(optimizer.minimize(cross_entropy), tf.assign_add(global_step, 1))

	return tf.contrib.learn.ModelFnOps(
		mode=mode, predictions=tf.nn.softmax(y), loss=cross_entropy, train_op=train)

N=1000
estimator = tf.contrib.learn.Estimator(model_fn=model)

#print(mnist.train.images.shape)
input_fn = tf.contrib.learn.io.numpy_input_fn(
	{'x':mnist.train.images}, mnist.train.labels, batch_size=100, num_epochs = 2)
test_input_fn = tf.contrib.learn.io.numpy_input_fn(
	{'x':mnist.test.images}, mnist.test.labels, batch_size=100, num_epochs = 2)
estimator.fit(input_fn=input_fn, steps=N)

train_loss = estimator.evaluate(input_fn=input_fn)
print("train loss %r" % train_loss)
test_loss = estimator.evaluate(input_fn=test_input_fn)
print("test loss %r" % test_loss)

pred_input_fn = tf.contrib.learn.io.numpy_input_fn({'x':mnist.test.images}, None)
prediction = np.array(list(estimator.predict(input_fn=pred_input_fn)))
print(prediction.shape)
print(mnist.test.labels.shape)
correct = tf.equal(tf.argmax(mnist.test.labels, 1), tf.argmax(prediction, 1))
print(correct.shape)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

sess = tf.Session()
print(sess.run(accuracy))
