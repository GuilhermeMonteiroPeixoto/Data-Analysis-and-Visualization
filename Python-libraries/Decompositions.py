import numpy as np

#Obtenha o L tri-triangular inferior na decomposição de Cholesky
x = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.int32)
L = np.linalg.cholesky(x)
print(L)
assert np.array_equal(np.dot(L, L.T.conjugate()), x)

#fatoração qr
x = np.array([[132, -11, 5], [9, 167, -658], [-9, 204, -49]], dtype=np.float32)
q, r = np.linalg.qr(x)
print ("q=", q, "\n\nr=", r)
assert np.allclose(np.dot(q, r), x)

#Fator x pela decomposição de valores singulares
x = np.array([[1, 0, 0, 0, 2], [0, 0, 9, 0, 0], [0, 0, 1, 0, 0], [0, 7, 0, 0, 0]], dtype=np.float32)
U, s, V = np.linalg.svd(x, full_matrices=False)
print ("U=", U, "\n\ns=", s, "\n\nV=", V)
assert np.allclose(np.dot(U, np.dot(np.diag(s), V)), x)