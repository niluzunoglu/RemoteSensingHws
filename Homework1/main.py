import numpy as np
import matplotlib.pyplot as plt

def manual_convolve(x, h):
    len_y = len(x) + len(h) - 1
    y = np.zeros(len_y)
    
    for n in range(len_y):
        for k in range(len(x)):
            if 0 <= n - k < len(h):
                y[n] += x[k] * h[n - k]
    
    return y

# x[n] ve h[n] dizileri
x1 = np.array([1, 2, 3, 4])
h1 = np.array([0.2, 0.5, 0.2])

y1_manual = manual_convolve(x1, h1)
y1_numpy = np.convolve(x1, h1, mode='full')

plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.stem(x1)
plt.title('x[n]')

plt.subplot(2, 2, 2)
plt.stem(h1)
plt.title('h[n]')

plt.subplot(2, 2, 3)
plt.stem(y1_manual)
plt.title('Manual Convolution')

plt.subplot(2, 2, 4)
plt.stem(y1_numpy)
plt.title('Numpy Convolve')

plt.tight_layout()
plt.show()
