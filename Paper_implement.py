import numpy as np
import matplotlib.pyplot as plt

# =========================
# Physical & Simulation Parameters
# =========================
wavelength = 1064e-9       # meters
k = 2 * np.pi / wavelength
w0 = 5e-3                  # beam waist (m)
beam_spacing = 12e-3       # separation (m)
P = 1.0                    # power per beam (arb.)
Npix = 256                 # simulation grid
L = 0.12                   # physical window size (m)
dx = L / Npix
z_prop = 5.0               # propagation distance (m)

# =========================
# Spatial Grid
# =========================
x = np.linspace(-L/2, L/2, Npix)
y = np.linspace(-L/2, L/2, Npix)
X, Y = np.meshgrid(x, y)

# =========================
# Hexagonal Beam Array
# =========================
def hexagonal_array(n_rings, spacing):
    coords = [(0.0, 0.0)]
    for r in range(1, n_rings + 1):
        for k in range(6):
            angle = k * np.pi / 3
            for i in range(r):
                x = spacing * (r * np.cos(angle) - i * np.cos(angle + np.pi / 3))
                y = spacing * (r * np.sin(angle) - i * np.sin(angle + np.pi / 3))
                coords.append((x, y))
    return np.array(coords)

beam_centers = hexagonal_array(n_rings=3, spacing=beam_spacing)
Nbeams = len(beam_centers)

# =========================
# Near-field Electric Field
# =========================
def near_field(phases):
    E = np.zeros_like(X, dtype=complex)
    for (xn, yn), phi in zip(beam_centers, phases):
        r2 = (X - xn)**2 + (Y - yn)**2
        E += np.sqrt(2*P/(np.pi*w0**2)) * np.exp(-r2/w0**2) * np.exp(-1j*phi)
    return E

# =========================
# Angular Spectrum Propagation
# =========================
def angular_spectrum(E):
    fx = np.fft.fftfreq(Npix, dx)
    fy = np.fft.fftfreq(Npix, dx)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * k * z_prop * np.sqrt(
        1 - (wavelength*FX)**2 - (wavelength*FY)**2
    ))
    E_f = np.fft.fft2(E)
    return np.fft.ifft2(E_f * H)

# =========================
# Power-in-Bucket (PIB)
# =========================
def power_in_bucket(I, cx, cy, r=5):
    Yg, Xg = np.ogrid[:Npix, :Npix]
    mask = (Xg - cx)**2 + (Yg - cy)**2 <= r**2
    return I[mask].sum() / I.sum()

# =========================
# Merit Function
# =========================
def merit(phases, target):
    E = near_field(phases)
    Ef = angular_spectrum(E)
    I = np.abs(Ef)**2
    return power_in_bucket(I, target[0], target[1])

# =========================
# Gradient Estimation (Finite Difference)
# =========================
def gradient(phases, target, delta=1e-3):
    g = np.zeros_like(phases)
    J0 = merit(phases, target)
    for i in range(len(phases)):
        dp = phases.copy()
        dp[i] += delta
        g[i] = (merit(dp, target) - J0) / delta
    return g

# =========================
# Adagrad Optimizer
# =========================
def adagrad(
    phases,
    target,
    alpha=0.2,
    eta=0.01,
    eps=1e-8,
    iters=150
):
    G = np.zeros_like(phases)
    history = []

    for t in range(1, iters+1):
        g = gradient(phases, target)
        G += g**2
        alpha_t = alpha / (1 + (t-1)*eta)
        phases += alpha_t * g / (np.sqrt(G) + eps)
        history.append(merit(phases, target))

        if t % 10 == 0:
            print(f"Iter {t:3d} | PIB = {history[-1]:.4f}")

    return phases, history

# =========================
# Run Beam Steering
# =========================
np.random.seed(1)
phases0 = np.random.uniform(0, 2*np.pi, Nbeams)
target_pixel = (148, 148)

phases_opt, pib_history = adagrad(phases0, target_pixel)

# =========================
# Visualization
# =========================
Ef_final = angular_spectrum(near_field(phases_opt))
I_final = np.abs(Ef_final)**2

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.imshow(I_final, cmap='inferno')
plt.scatter(*target_pixel, color='cyan')
plt.title("Far-field Intensity (Steered)")
plt.colorbar()

plt.subplot(1,2,2)
plt.plot(pib_history)
plt.xlabel("Iteration")
plt.ylabel("PIB")
plt.title("Adagrad Convergence")
plt.tight_layout()
plt.show()
