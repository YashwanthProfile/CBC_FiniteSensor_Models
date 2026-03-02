import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Physical & Simulation Parameters
# =====================================================
wavelength = 1064e-9
k = 2 * np.pi / wavelength
w0 = 5e-3
beam_spacing = 12e-3
P = 1.0
Npix = 256
L = 0.12
dx = L / Npix
z_prop = 100.0

# =====================================================
# Spatial Grid
# =====================================================
x = np.linspace(-L/2, L/2, Npix)
y = np.linspace(-L/2, L/2, Npix)
X, Y = np.meshgrid(x, y)

# =====================================================
# Hexagonal Beam Array
# =====================================================
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

# =====================================================
# Near-field
# =====================================================
def near_field(phases):
    E = np.zeros_like(X, dtype=complex)
    for (xn, yn), phi in zip(beam_centers, phases):
        r2 = (X-xn)**2 + (Y-yn)**2
        E += np.sqrt(2*P/(np.pi*w0**2)) * np.exp(-r2/w0**2) * np.exp(-1j*phi)
    return E

# =====================================================
# Angular Spectrum
# =====================================================
def angular_spectrum(E):
    fx = np.fft.fftfreq(Npix, dx)
    fy = np.fft.fftfreq(Npix, dx)
    FX, FY = np.meshgrid(fx, fy)
    arg = 1 - (wavelength*FX)**2 - (wavelength*FY)**2
    arg[arg < 0] = 0
    H = np.exp(1j * k * z_prop * np.sqrt(arg))
    return np.fft.ifft2(np.fft.fft2(E) * H)

# =====================================================
# Power-in-Bucket (TRUE)
# =====================================================
def power_in_bucket(I, cx, cy, r=6):
    Yg, Xg = np.ogrid[:Npix, :Npix]
    mask = (Xg-cx)**2 + (Yg-cy)**2 <= r**2
    return I[mask].sum() / I.sum()

# =====================================================
# Gaussian Sensor
# =====================================================
def gaussian_sensor(I, centers, sigma=6):
    Yg, Xg = np.indices(I.shape)
    y = np.zeros(len(centers))
    for i, (cx, cy) in enumerate(centers):
        w = np.exp(-((Xg-cx)**2 + (Yg-cy)**2)/(2*sigma**2))
        y[i] = np.sum(I*w)
    return y

# =====================================================
# Merit functions
# =====================================================
def merit_full(phases, target):
    I = np.abs(angular_spectrum(near_field(phases)))**2
    return power_in_bucket(I, *target)

def merit_gaussian(phases, centers, tidx):
    I = np.abs(angular_spectrum(near_field(phases)))**2
    y = gaussian_sensor(I, centers)
    return y[tidx] / y.sum()

# =====================================================
# Gradient + Adagrad (stores phase history)
# =====================================================
def gradient(phases, merit_fn, delta=1e-3):
    g = np.zeros_like(phases)
    J0 = merit_fn(phases)
    for i in range(len(phases)):
        dp = phases.copy()
        dp[i] += delta
        g[i] = (merit_fn(dp) - J0) / delta
    return g

def adagrad(phases0, merit_fn, iters=30):
    phases = phases0.copy()
    G = np.zeros_like(phases)
    hist = []
    phase_hist = []

    for t in range(iters):
        g = gradient(phases, merit_fn)
        G += g**2
        phases += 0.2/(1+t*0.01) * g / (np.sqrt(G) + 1e-8)

        hist.append(merit_fn(phases))
        phase_hist.append(phases.copy())

        if t % 10 == 0:
            print(f"Iter {t:3d} | Objective = {hist[-1]:.4f}")

    return phases, np.array(hist), np.array(phase_hist)

# =====================================================
# Far-field support for sensor placement
# =====================================================
def far_field_support(radius_energy=0.99):
    ph = np.random.uniform(0, 2*np.pi, Nbeams)
    I = np.abs(angular_spectrum(near_field(ph)))**2
    flat = I.flatten()
    idx = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[idx]) / flat.sum()
    cutoff = flat[idx][np.searchsorted(cum, radius_energy)]
    return I >= cutoff

support_mask = far_field_support()

# =====================================================
# Sensor templates (constrained)
# =====================================================
target_pixel = (148, 148)
support_indices = np.column_stack(np.where(support_mask))

def sensors_random(M):
    idx = np.random.choice(len(support_indices), M, replace=False)
    s = support_indices[idx][:, ::-1]
    s[0] = target_pixel
    return [tuple(p) for p in s]

def sensors_grid(M):
    # Define grid resolution
    n = int(np.ceil(np.sqrt(M)))

    # Bounding box of support
    ys, xs = np.where(support_mask)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    # Generate grid
    gx = np.linspace(xmin, xmax, n).astype(int)
    gy = np.linspace(ymin, ymax, n).astype(int)

    grid = [(x, y) for x in gx for y in gy]

    # Keep only points inside support
    grid = [p for p in grid if support_mask[p[1], p[0]]]

    # Truncate
    grid = grid[:M]

    # Ensure target is included
    if target_pixel not in grid:
        grid[0] = target_pixel

    return grid


def sensors_laser_matched():
    f = lambda u: int((u+L/2)/L*Npix)
    s = [(f(x), f(y)) for x, y in beam_centers]
    s = [p for p in s if support_mask[p[1], p[0]]]
    s[0] = target_pixel
    return s

sensor_sets = {
    "Full Image": None,
    "Random": sensors_random(Nbeams),
    "Grid": sensors_grid(Nbeams),
    "Laser-Matched": sensors_laser_matched()
}

# =====================================================
# Run experiments
# =====================================================
np.random.seed(1)
ph0 = np.random.uniform(0, 2*np.pi, Nbeams)
results = {}

for name, sensors in sensor_sets.items():
    print(f"\n--- {name} optimization ---")

    if name == "Full Image":
        ph, hist, ph_hist = adagrad(ph0, lambda p: merit_full(p, target_pixel))
    else:
        ph, hist, ph_hist = adagrad(ph0, lambda p, s=sensors: merit_gaussian(p, s, 0))

    true_pib_hist = []
    for p in ph_hist:
        I = np.abs(angular_spectrum(near_field(p)))**2
        true_pib_hist.append(power_in_bucket(I, *target_pixel))

    I_final = np.abs(angular_spectrum(near_field(ph)))**2
    pib_final = power_in_bucket(I_final, *target_pixel)

    results[name] = dict(
        ph=ph,
        I=I_final,
        sensors=sensors,
        obj_hist=hist,
        true_pib_hist=np.array(true_pib_hist),
        pib=pib_final
    )

# =====================================================
# Print TRUE PIB
# =====================================================
print("\n========== TRUE PIB (Full Image Evaluation) ==========")
for name, r in results.items():
    print(f"{name:15s}: {r['pib']:.6f}")
print("====================================================")

# =====================================================
# Figure A: Near-field + all Far-fields
# =====================================================
plt.figure(figsize=(12,10))

plt.subplot(2,3,1)
plt.imshow(np.abs(near_field(ph0))**2, cmap='inferno')
plt.title("Near-field Intensity")
plt.colorbar()

i = 2
for name, r in results.items():
    plt.subplot(2,3,i)
    plt.imshow(r["I"], cmap='inferno')
    plt.scatter(*target_pixel, c='cyan', marker='*', s=80)
    if r["sensors"] is not None:
        xs, ys = zip(*r["sensors"])
        plt.scatter(xs, ys, facecolors='none', edgecolors='white')
    plt.title(f"Far-field ({name})")
    i += 1

plt.tight_layout()
plt.show()

# =====================================================
# Figure B: Sensor Objective Convergence
# =====================================================
plt.figure()
for name, r in results.items():
    plt.plot(r["obj_hist"], label=name)
plt.xlabel("Iteration")
plt.ylabel("Objective")
plt.legend()
plt.grid(True)
plt.title("Sensor / PIB Objective Convergence")
plt.show()

# =====================================================
# Figure C: TRUE PIB Convergence
# =====================================================
plt.figure()
for name, r in results.items():
    plt.plot(r["true_pib_hist"], label=name)
plt.xlabel("Iteration")
plt.ylabel("TRUE PIB")
plt.legend()
plt.grid(True)
plt.title("True PIB Convergence (Full Image Evaluation)")
plt.show()
