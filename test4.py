import tensorflow as tf
import numpy as np

def model(features, labels, mode):
	m = tf.get_variable('w', [1], dtype=tf.float64)
	b = tf.get_variable('b', [1], dtype=tf.float64)
	print(m)
	print(b)
	y = m*features['x'] + b

	loss = tf.reduce_sum(tf.square(y-labels))
	global_step = tf.train.get_global_step()
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
	return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)


#features = [tf.contrib.layers.real_valued_column('x', dimension=1)]
#estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
estimator = tf.contrib.learn.Estimator(model_fn=model)

N = 1000
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7., 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({'x':x_train}, y_train, batch_size=4, num_epochs=N)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({'x':x_eval}, y_eval, batch_size=4, num_epochs=N)

estimator.fit(input_fn=input_fn, steps=N)

train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r" % train_loss)
print("eval loss: %r" % eval_loss)
