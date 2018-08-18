import tensorflow as tf

# print(tensorflow.__version__)

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y+y+2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()

x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())
    print(z.eval())

with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)
    
