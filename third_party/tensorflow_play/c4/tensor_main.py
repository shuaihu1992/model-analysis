import tensorflow as tf
import numpy as np


from tensorflow import SparseTensor

from tensorflow import Tensor

## Constant 值不能改变
# https://github.com/tensorflow/docs/blob/r1.4/site/en/api_docs/api_docs/python/tf/constant.md

# Constant 1-D Tensor populated with value list.
tensor = tf.constant([1, 2, 3, 4, 5, 6, 7])

# Constant 2-D tensor populated with scalar value -1.
tensor = tf.constant(-1.0, shape=[2, 3])

print(tensor)

print(type(tensor))


## variable 值可以改变的一种

# https://github.com/tensorflow/docs/blob/r1.4/site/en/api_docs/api_docs/python/tf/Variable.md

# Create a variable.
var = tf.Variable(3)
var = tf.Variable(3,dtype=tf.int32)



# Use the variable in the graph like any Tensor.
# y = tf.matmul(w, ...another variable or tensor...)
#
# The overloaded operators are available too.
# z = tf.sigmoid(w + y)
#
# Assign a new value to the variable with `assign()` or a related method.
# w.assign(w + 1.0)
# w.assign_add(1.0)

## placeholder 占位符

x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.


## 稀疏张量 矩阵
# https://github.com/tensorflow/docs/blob/r1.4/site/en/api_docs/api_docs/python/tf/SparseTensor.md


st = SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

print(st)
