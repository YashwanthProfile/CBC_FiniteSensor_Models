import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import os

# =====================================================
# Experiment metadata
# =====================================================
today = date.today().isoformat()
outdir = "results"
os.makedirs(outdir, exist_ok=True)

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
        r2 = (X - xn)**2 + (Y - yn)**2
        E += np.sqrt(2*P/(np.pi*w0**2)) * np.exp(-r2/w0**2) * np.exp(-1j*phi)
    return E

# =====================================================
# Angular Spectrum Propagation
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
# TRUE Power-in-Bucket
# =====================================================
def power_in_bucket(I, cx, cy, r=6):
    Yg, Xg = np.ogrid[:Npix, :Npix]
    mask = (Xg - cx)**2 + (Yg - cy)**2 <= r**2
    return I[mask].sum() / I.sum()

# =====================================================
# Gaussian Sensor Model
# =====================================================
def gaussian_sensor(I, centers, sigma=6):
    Yg, Xg = np.indices(I.shape)
    y = np.zeros(len(centers))
    for i, (cx, cy) in enumerate(centers):
        w = np.exp(-((Xg - cx)**2 + (Yg - cy)**2)/(2*sigma**2))
        y[i] = np.sum(I * w)
    return y

# =====================================================
# Merit Functions
# =====================================================
def merit_full(phases, targets):
    I = np.abs(angular_spectrum(near_field(phases)))**2
    return np.mean([power_in_bucket(I, cx, cy) for cx, cy in targets])

def merit_gaussian(phases, centers, targets, sigma=6):
    if len(centers) == 0 or len(targets) == 0:
        return 0.0

    I = np.abs(angular_spectrum(near_field(phases)))**2
    y = gaussian_sensor(I, centers, sigma=sigma)

    if y.sum() == 0:
        return 0.0

    Yg, Xg = np.indices(I.shape)
    J = 0.0
    for cx, cy in targets:
        w = np.exp(-((Xg - cx)**2 + (Yg - cy)**2)/(2*sigma**2))
        J += np.sum(I * w)

    return J / y.sum()

# =====================================================
# Optimization (Adagrad + Finite Difference)
# =====================================================
def gradient(phases, merit_fn, delta=1e-3):
    g = np.zeros_like(phases)
    J0 = merit_fn(phases)
    for i in range(len(phases)):
        dp = phases.copy()
        dp[i] += delta
        g[i] = (merit_fn(dp) - J0) / delta
    return g

def adagrad(phases0, merit_fn, iters=80):
    phases = phases0.copy()
    G = np.zeros_like(phases)
    obj_hist = []
    phase_hist = []

    for t in range(iters):
        g = gradient(phases, merit_fn)
        G += g**2
        phases += 0.2/(1 + 0.01*t) * g / (np.sqrt(G) + 1e-8)
        obj_hist.append(merit_fn(phases))
        phase_hist.append(phases.copy())

    return phases, np.array(obj_hist), np.array(phase_hist)

# =====================================================
# Far-field Support for Sensor Placement
# =====================================================
def far_field_support(energy_frac=0.99):
    ph = np.random.uniform(0, 2*np.pi, Nbeams)
    I = np.abs(angular_spectrum(near_field(ph)))**2
    flat = I.flatten()
    idx = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[idx]) / flat.sum()
    cutoff = flat[idx][np.searchsorted(cum, energy_frac)]
    return I >= cutoff

support_mask = far_field_support()
support_points = np.column_stack(np.where(support_mask))[:, ::-1]

# =====================================================
# Sensor Placement Templates
# =====================================================
def sensors_random(M):
    idx = np.random.choice(len(support_points), M, replace=False)
    return [tuple(p) for p in support_points[idx]]

def sensors_grid(M):
    n = int(np.ceil(np.sqrt(M)))

    ys, xs = np.where(support_mask)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    gx = np.linspace(xmin, xmax, n).astype(int)
    gy = np.linspace(ymin, ymax, n).astype(int)

    grid = [(x, y) for x in gx for y in gy]
    grid = [p for p in grid if support_mask[p[1], p[0]]]

    if len(grid) < M:
        return sensors_random(M)

    return grid[:M]

# =====================================================
# Experiment Sweep
# =====================================================
np.random.seed(1)

targets = [(148, 148)]
sensor_counts = [4, 8, 12, 16, 24, 32]
final_true_pib = {"Full": [], "Random": [], "Grid": []}

for M in sensor_counts:
    print(f"\n=== Sensors: {M} | z = {z_prop} m ===")

    ph0 = np.random.uniform(0, 2*np.pi, Nbeams)

    sensors_r = sensors_random(M)
    sensors_g = sensors_grid(M)

    ph_f, h_f, phh_f = adagrad(ph0, lambda p: merit_full(p, targets))
    ph_r, h_r, phh_r = adagrad(ph0, lambda p: merit_gaussian(p, sensors_r, targets))
    ph_g, h_g, phh_g = adagrad(ph0, lambda p: merit_gaussian(p, sensors_g, targets))

    def true_hist(ph_hist):
        out = []
        for p in ph_hist:
            I = np.abs(angular_spectrum(near_field(p)))**2
            out.append(np.mean([power_in_bucket(I, cx, cy) for cx, cy in targets]))
        return np.array(out)

    t_f = true_hist(phh_f)
    t_r = true_hist(phh_r)
    t_g = true_hist(phh_g)

    final_true_pib["Full"].append(t_f[-1])
    final_true_pib["Random"].append(t_r[-1])
    final_true_pib["Grid"].append(t_g[-1])

    # ================= Figure 1 =================
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.imshow(np.abs(near_field(ph0))**2, cmap="inferno")
    plt.title("Near-field")
    plt.colorbar()

    for i,(name,ph,sensors) in enumerate([
        ("Full Image", ph_f, None),
        ("Random", ph_r, sensors_r),
        ("Grid", ph_g, sensors_g)
    ], start=2):
        plt.subplot(2,2,i)
        I = np.abs(angular_spectrum(near_field(ph)))**2
        plt.imshow(I, cmap="inferno")
        for cx, cy in targets:
            plt.scatter(cx, cy, c="cyan", marker="*", s=80)
        if sensors:
            xs, ys = zip(*sensors)
            plt.scatter(xs, ys, facecolors="none", edgecolors="white")
        plt.title(name)

    plt.suptitle(f"z={z_prop} m | Sensors={M}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/fields_z{z_prop}_M{M}_{today}.png", dpi=200)
    plt.close()

    # ================= Figure 2 =================
    plt.figure()
    plt.plot(h_f, label="Full Image")
    plt.plot(h_r, label="Random")
    plt.plot(h_g, label="Grid")
    plt.legend(); plt.grid()
    plt.xlabel("Iteration"); plt.ylabel("Objective")
    plt.title(f"Sensor Objective | z={z_prop} m | M={M}")
    plt.savefig(f"{outdir}/objective_z{z_prop}_M{M}_{today}.png", dpi=200)
    plt.close()

    # ================= Figure 3 =================
    plt.figure()
    plt.plot(t_f, label="Full Image")
    plt.plot(t_r, label="Random")
    plt.plot(t_g, label="Grid")
    plt.legend(); plt.grid()
    plt.xlabel("Iteration"); plt.ylabel("TRUE PIB")
    plt.title(f"TRUE PIB | z={z_prop} m | M={M}")
    plt.savefig(f"{outdir}/true_pib_z{z_prop}_M{M}_{today}.png", dpi=200)
    plt.close()

# =====================================================
# Final Summary Plot
# =====================================================
plt.figure()
for k, v in final_true_pib.items():
    plt.plot(sensor_counts, v, "-o", label=k)
plt.xlabel("Number of sensors")
plt.ylabel("Final TRUE PIB")
plt.legend(); plt.grid()
plt.title(f"Final TRUE PIB vs Sensors | z={z_prop} m")
plt.savefig(f"{outdir}/final_pib_vs_M_z{z_prop}_{today}.png", dpi=200)
plt.show()
