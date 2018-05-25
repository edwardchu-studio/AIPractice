import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

x=tf.placeholder(tf.float32,shape=[None,28,28])
z=tf.Variable(tf.truncated_normal([28,28,1]))

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

x=tf.placeholder(tf.float32,shape=[None,28,28])
z=tf.Variable(tf.truncated_normal([28,28,1]))

