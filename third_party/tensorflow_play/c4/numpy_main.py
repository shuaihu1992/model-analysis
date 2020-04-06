import numpy as np


vector = np.array([1,2,3])
vector.shape

vector.ndim

type(vector)

matrix = np.array([[1,2],[3,4]])

matrix.shape

matrix.ndim

matrix.size

one = np.arange(12)

two = one.reshape((3,4))

#对角线矩阵

zeros = np.zeros((3,4))

print(zeros)

ones = np.ones((5,6))

print(ones)

ident = np.eye(4)

print(ident)


