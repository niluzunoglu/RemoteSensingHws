import numpy as np
import matplotlib.pyplot as plt

Omega = np.arange(-2*np.pi, 2*np.pi, 0.1)

N_values = [3, 6, 9, 21]
fig, axs = plt.subplots(len(N_values), 2, figsize=(10, 12))
fig.suptitle('DTFT H(e^{jΩ}) için Genlik ve Faz Spektrumu')

for idx, N in enumerate(N_values):

    # H(e^{jΩ}) = (1 - e^{-jΩN}) / (1 - e^{-jΩ})
    
    numerator   = 1 - np.exp(-1j * Omega * N)
    denominator = 1 - np.exp(-1j * Omega)
    
    epsilon = 1e-12
    denominator_safe = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
    
    H = numerator / denominator_safe
    
    magnitude = np.abs(H)
    phase     = np.angle(H)
    
    axs[idx, 0].plot(Omega, magnitude, label=f'N={N}')
    axs[idx, 0].set_title(f'N={N} - Genlik')
    axs[idx, 0].set_xlabel('Ω (radyan)')
    axs[idx, 0].set_ylabel('|H(e^{jΩ})|')
    axs[idx, 0].grid(True)
    
    axs[idx, 1].plot(Omega, phase, label=f'N={N}')
    axs[idx, 1].set_title(f'N={N} - Faz')
    axs[idx, 1].set_xlabel('Ω (radyan)')
    axs[idx, 1].set_ylabel('∠H(e^{jΩ}) (radyan)')
    axs[idx, 1].grid(True)

plt.tight_layout()
plt.savefig('DTFT_H_spectra.png')
plt.show()
