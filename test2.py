import tensorflow as tf
m = tf.Variable([0.3], dtype=tf.float64)
b = tf.Variable([-0.3], dtype=tf.float64)
x = tf.placeholder(tf.float64)

linear = m*x + b

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
y = tf.placeholder(tf.float64)
squared_deltas = tf.square(linear - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0, -1, -2, -3]}))

M = tf.assign(m, [-1])
B = tf.assign(b, [1])
sess.run([M, B])

print(sess.run(loss, {x:[1,2,3,4], y:[0, -1, -2, -3]}))
