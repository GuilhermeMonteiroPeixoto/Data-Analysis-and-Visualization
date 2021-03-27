from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from datetime import date
date.today()

a = np.array([1+2j, 3+4j, 5+6j])
real = a.real
imag = a.imag
print("real part=", real)
print("imaginary part=", imag)

a = np.array([1+2j, 3+4j, 5+6j])
a.real = 9
a.imag = [5, 7, 9]
print(a)

a = 1+2j
output = np.conjugate(a)
print(output)