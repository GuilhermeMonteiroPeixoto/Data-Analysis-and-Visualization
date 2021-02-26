import numpy as np

#Frobenius norm
x = np.arange(0, 16).reshape((4, 4))
print (np.linalg.norm(x, 'fro'))

#determinant
x = np.arange(1, 5).reshape((2, 2))
out1 = np.linalg.det(x)
out2 = x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]
assert np.allclose(out1, out2)
print (out1)

#classificação de x
x = np.eye(4)
out1 = np.linalg.matrix_rank(x)
out2 = np.linalg.svd(x)[1].size
assert out1 == out2
print (out1)

#sinal e o logaritmo natural do determinante de x
x = np.arange(1, 5).reshape((2, 2))
sign, logdet = np.linalg.slogdet(x)
det = np.linalg.det(x)
assert sign == np.sign(det)
assert logdet == np.log(np.abs(det))
print (sign, logdet)

#soma ao longo da diagonal de x
x = np.eye(4)
out1 = np.trace(x)
out2 = x.diagonal().sum()
assert out1 == out2
print (out1)