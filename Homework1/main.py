import numpy as np
import matplotlib.pyplot as plt

def find_low_and_high(k_array):
    low = np.min(k_array)
    high = np.max(k_array)
    return low, high
    
def my_convolution(x_n, h_n):

    x_low, x_high = find_low_and_high(x_n)
    h_low, h_high = find_low_and_high(h_n)
        
    y_low  = x_low  + h_low
    y_high = x_high + h_high

    print("x[n] dizisi aralığı: ", x_low, " ile ", x_high)
    print("h[n] dizisi aralığı: ", h_low, " ile ", h_high)
    print("y[n] dizisi aralığı: ", y_low, " ile ", y_high)

    len_y = len(x_n) + len(h_n) - 1
    y = np.zeros(len_y)
    
    # Convolution işlemi 
    for n in range(len_y):
        for k in range(len(x_n)):
            if 0 <= n - k < len(h_n):
                y[n] += x_n[k] * h_n[n - k]
    
    return y

def draw_plot(location, input, title):
    plt.subplot(location)
    plt.stem(input)
    plt.title(title)

if __name__ == "__main__":

    # ------------------------------------------------- #

    # x[n] ve h[n] dizileri
    x1 = np.array([1, 2, 3, 4])
    h1 = np.array([0.2, 0.5, 0.2])
    y1_manual = my_convolution(x1, h1)
    y1_numpy = np.convolve(x1, h1, mode='full')

    plt.figure(figsize=(6, 6))
    draw_plot(221, x1, 'x[n]')
    draw_plot(222, h1, 'h[n]')
    draw_plot(223, y1_manual, 'My Convolution Function')
    draw_plot(224, y1_numpy, 'Numpy Convolve')
    plt.tight_layout()
    plt.savefig("ornek1.png")
    plt.show()

    # ------------------------------------------------- #

    # x[n] ve h[n] dizileri
    x2 = np.array([1,  2,  1, 0, -1])
    h2 = np.array([1, -1,  2])
    y2_manual = my_convolution(x2, h2)
    y2_numpy = np.convolve(x2, h2, mode='full')

    plt.figure(figsize=(6, 6))
    draw_plot(221, x2, 'x[n]')
    draw_plot(222, h2, 'h[n]')
    draw_plot(223, y2_manual, 'My Convolution Function')
    draw_plot(224, y2_numpy, 'Numpy Convolve')
    plt.tight_layout()
    plt.savefig("ornek2.png")
    plt.show()

    # ------------------------------------------------- #

    # x[n] ve h[n] dizileri
    x3 = np.array([1, 1, 1, 1, 1])
    h3 = np.array([0, 1, 0.5])
    y3_manual = my_convolution(x3, h3)
    y3_numpy = np.convolve(x3, h3, mode='full')

    plt.figure(figsize=(6, 6))
    draw_plot(221, x3, 'x[n]')
    draw_plot(222, h3, 'h[n]')
    draw_plot(223, y3_manual, 'My Convolution Function')
    draw_plot(224, y3_numpy, 'Numpy Convolve')
    plt.tight_layout()
    plt.savefig("ornek3.png")
    plt.show()
