import numpy as np
import matplotlib.pyplot as plt

# =========================
# Physical & Simulation Parameters
# =========================
wavelength = 1064e-9
k = 2 * np.pi / wavelength
w0 = 5e-3
beam_spacing = 12e-3
P = 1.0
Npix = 256
L = 0.12
dx = L / Npix
z_prop = 5.0

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
                coords.append((
                    spacing * (r*np.cos(angle) - i*np.cos(angle + np.pi/3)),
                    spacing * (r*np.sin(angle) - i*np.sin(angle + np.pi/3))
                ))
    return np.array(coords)

beam_centers = hexagonal_array(3, beam_spacing)
Nbeams = len(beam_centers)

# =========================
# Near-field
# =========================
def near_field(phases):
    E = np.zeros_like(X, dtype=complex)
    for (xn, yn), phi in zip(beam_centers, phases):
        r2 = (X-xn)**2 + (Y-yn)**2
        E += np.sqrt(2*P/(np.pi*w0**2)) * np.exp(-r2/w0**2) * np.exp(-1j*phi)
    return E

# =========================
# Angular Spectrum Propagation
# =========================
def angular_spectrum(E):
    fx = np.fft.fftfreq(Npix, dx)
    fy = np.fft.fftfreq(Npix, dx)
    FX, FY = np.meshgrid(fx, fy)
    arg = 1 - (wavelength*FX)**2 - (wavelength*FY)**2
    arg[arg < 0] = 0
    H = np.exp(1j * k * z_prop * np.sqrt(arg))
    return np.fft.ifft2(np.fft.fft2(E) * H)

# =========================
# Full-image PIB
# =========================
def power_in_bucket(I, cx, cy, r=6):
    Yg, Xg = np.ogrid[:Npix, :Npix]
    mask = (Xg-cx)**2 + (Yg-cy)**2 <= r**2
    return I[mask].sum() / I.sum()

# =========================
# Sensor Models
# =========================
def gaussian_sensor(I, centers, sigma=6):
    Yg, Xg = np.indices(I.shape)
    y = np.zeros(len(centers))
    for i, (cx, cy) in enumerate(centers):
        w = np.exp(-((Xg-cx)**2 + (Yg-cy)**2)/(2*sigma**2))
        y[i] = np.sum(I*w)
    return y

def bitmap_sensor(I, masks):
    return np.array([I[m].sum() for m in masks])

# =========================
# Merit Functions
# =========================
def merit_full(phases, target):
    I = np.abs(angular_spectrum(near_field(phases)))**2
    return power_in_bucket(I, *target)

def merit_gaussian(phases, centers, tidx):
    I = np.abs(angular_spectrum(near_field(phases)))**2
    y = gaussian_sensor(I, centers)
    return y[tidx] / y.sum()

def merit_bitmap(phases, masks, tidx):
    I = np.abs(angular_spectrum(near_field(phases)))**2
    y = bitmap_sensor(I, masks)
    return y[tidx] / y.sum()

# =========================
# Gradient (generic merit)
# =========================
def gradient(phases, merit_fn, delta=1e-3):
    g = np.zeros_like(phases)
    J0 = merit_fn(phases)
    for i in range(len(phases)):
        dp = phases.copy()
        dp[i] += delta
        g[i] = (merit_fn(dp) - J0) / delta
    return g

# =========================
# Adagrad
# =========================
def adagrad(phases0, merit_fn, iters=150, alpha=0.2, eta=0.01):
    phases = phases0.copy()
    G = np.zeros_like(phases)
    hist = []

    for t in range(iters):
        g = gradient(phases, merit_fn)
        G += g**2
        phases += (alpha/(1+t*eta)) * g / (np.sqrt(G) + 1e-8)
        hist.append(merit_fn(phases))
        if t % 10 == 0:
            print(f"Iter {t:3d} | PIB = {hist[-1]:.4f}")
    return phases, np.array(hist)

# =========================
# Sensor Configuration (10 sensors)
# =========================
target_pixel = (148, 148)

sensor_centers = [
    target_pixel,
    (120,148),(176,148),(148,120),(148,176),
    (130,130),(166,130),(130,166),(166,166),(148,160)
]

def circular_mask(cx, cy, r=6):
    Yg, Xg = np.indices((Npix, Npix))
    return (Xg-cx)**2 + (Yg-cy)**2 <= r**2

bitmap_masks = [circular_mask(*c) for c in sensor_centers]

# =========================
# Run All Optimizations
# =========================
np.random.seed(1)
phases0 = np.random.uniform(0, 2*np.pi, Nbeams)

print("\n------ Starting Full Image Optimization ------")
ph_full, h_full = adagrad(phases0, lambda p: merit_full(p, target_pixel))
print("\n------ Starting Gaussian-Sensor Optimization ------")
ph_g, h_g = adagrad(phases0, lambda p: merit_gaussian(p, sensor_centers, 0))
print("\n------ Starting Bitmap-Sensor Optimization ------")
ph_b, h_b = adagrad(phases0, lambda p: merit_bitmap(p, bitmap_masks, 0))

# =========================
# Final Fields
# =========================
I_nf = np.abs(near_field(phases0))**2
I_full = np.abs(angular_spectrum(near_field(ph_full)))**2
I_g = np.abs(angular_spectrum(near_field(ph_g)))**2
I_b = np.abs(angular_spectrum(near_field(ph_b)))**2


# ======================================================
# POST-OPTIMIZATION: TRUE IMAGE-BASED PIB COMPARISON
# ======================================================

def true_pib_from_phases(phases, target_pixel):
    E = near_field(phases)
    Ef = angular_spectrum(E)
    I = np.abs(Ef)**2
    pib = power_in_bucket(I, target_pixel[0], target_pixel[1])
    return pib, I


pib_full_true, I_full_true = true_pib_from_phases(ph_full, target_pixel)
pib_gauss_true, I_gauss_true = true_pib_from_phases(ph_g, target_pixel)
pib_bitmap_true, I_bitmap_true = true_pib_from_phases(ph_b, target_pixel)

print("\n========== TRUE FULL-IMAGE PIB COMPARISON ==========")
print(f"Full-image optimized PIB   : {pib_full_true:.6f}")
print(f"Gaussian-sensor optimized PIB : {pib_gauss_true:.6f}")
print(f"Bitmap-sensor optimized PIB   : {pib_bitmap_true:.6f}")
print("===================================================\n")

# =========================
# Figure 1
# =========================
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(I_nf, cmap='inferno')
plt.title("Near-field Intensity")
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(I_full, cmap='inferno')
plt.title("Far-field (Full Image)")
plt.scatter(*target_pixel, c='cyan')

plt.subplot(2,2,3)
plt.imshow(I_g, cmap='inferno')
plt.title("Far-field (Gaussian Sensors)")
plt.scatter(*target_pixel, c='cyan')

plt.subplot(2,2,4)
plt.imshow(I_b, cmap='inferno')
plt.title("Far-field (Bitmap Sensors)")
plt.scatter(*target_pixel, c='cyan')

plt.tight_layout()
# plt.show()

# =========================
# Figure 2
# =========================
plt.figure()
plt.plot(h_full, label="Full Image")
plt.plot(h_g, label="Gaussian Sensors")
plt.plot(h_b, label="Bitmap Sensors")
plt.xlabel("Iteration")
plt.ylabel("PIB")
plt.legend()
plt.title("PIB Convergence Comparison")
plt.grid(True)
plt.show()