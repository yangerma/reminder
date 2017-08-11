from __future__ import print_function
import tensorflow as tf

vecX = [1, 2, 3, 4]
vecY = [0, -1, -2, -3]
m = tf.Variable([0.3], dtype=tf.float64)
b = tf.Variable([-0.3], dtype=tf.float64)
x = tf.placeholder(tf.float64)
linear = m*x + b

y = tf.placeholder(tf.float64)
squared_deltas = tf.square(linear - y)
loss = tf.reduce_sum(squared_deltas)

rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

N = 500
step = 50
for i in range(N):
	sess.run(train, {x:vecX, y:vecY})
	if (i+1)%step == 0:
		print(sess.run([m, b]), end=" ||| loss=")
		print(sess.run(loss, {x:vecX, y:vecY}))

print("\nFinal: " + str(sess.run([m, b])))
