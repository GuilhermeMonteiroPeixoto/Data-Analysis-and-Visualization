from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

#Calcule o DFT unidimensional de a
a = np.exp(2j * np.pi * np.arange(8))
output = np.fft.fft(a)
print(output)

#calcule o DFT inverso unidimensional
print("a=", a)
inversed = np.fft.ifft(output)
print("inversed=", a)

#Transformada de Fourier discreta unidimensional para entrada real
a = [0, 1, 0, 0]
output = np.fft.rfft(a)
print(output)
assert output.size==len(a)//2+1 if len(a)%2==0 else (len(a)+1)//2

output2 = np.fft.fft(a)
print(output2)

inversed = np.fft.ifft(output)
print("inversed=", a)

#Retorne as frequências de amostra DFT de a
signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=np.float32)
fourier = np.fft.fft(signal)
n = signal.size
freq = np.fft.fftfreq(n, d=1)
print(freq)