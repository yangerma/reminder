import tensorflow as tf
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b
triple_node = adder_node * 3
sess = tf.Session()
print(sess.run(triple_node, {a:3, b:4.5}))
print(sess.run(triple_node, {a:3, b:[2,4]}))
'''
node3 = tf.add(node1, node2)
sess = tf.Session()
print("node3:", node3)
print(sess.run(node3))
'''

