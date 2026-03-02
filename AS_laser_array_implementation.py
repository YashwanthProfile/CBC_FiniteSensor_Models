import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Physical Parameters
# ============================================================
wavelength = 1064e-9       # Laser wavelength (m)
k = 2 * np.pi / wavelength
w0 = 5e-3                  # Beam waist (m)
P = 1.0                    # Power per beam (arb. units)

# ============================================================
# Simulation Grid
# ============================================================
N = 1000 #256                    # Grid size
L = 0.12                   # Physical window (m)
dx = L / N

x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# ============================================================
# Hexagonal Beam Geometry
# ============================================================
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

beam_spacing = 12e-3
beam_centers = hexagonal_array(n_rings=4, spacing=beam_spacing)
Nbeams = len(beam_centers)

# ============================================================
# Gaussian CBC Near Field (Eq. 1–2)
# ============================================================
def cbc_near_field(phases):
    """
    phases: array of length Nbeams (radians)
    """
    E = np.zeros((N, N), dtype=np.complex128)

    for (xn, yn), phi in zip(beam_centers, phases):
        r2 = (X - xn)**2 + (Y - yn)**2
        E += np.sqrt(2 * P / (np.pi * w0**2)) \
             * np.exp(-r2 / w0**2) \
             * np.exp(-1j * phi)

    return E

# ============================================================
# Angular Spectrum Propagation (Band-Limited)
# ============================================================
def angular_spectrum_propagation(E0, z):
    """
    E0: complex near field
    z : propagation distance (m)
    """
    fx = np.fft.fftfreq(N, dx)
    fy = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fy)

    # Band-limited transfer function
    root = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    root[root < 0] = 0  # suppress evanescent waves

    H = np.exp(1j * k * z * np.sqrt(root))

    E0_f = np.fft.fft2(E0)
    Ez = np.fft.ifft2(E0_f * H)

    return Ez

# ============================================================
# Example Usage
# ============================================================
if __name__ == "__main__":

    # Arbitrary phase distribution (user-defined)
    np.random.seed(0)
    phases = np.random.uniform(0, 2*np.pi, Nbeams)

    # Propagation distance (meters)
    z = 100.0

    # Compute fields
    E_near = cbc_near_field(phases)
    E_far = angular_spectrum_propagation(E_near, z)

    # Intensities
    I_near = np.abs(E_near)**2
    I_far = np.abs(E_far)**2

    # ========================================================
    # Visualization
    # ========================================================
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(I_near, cmap='jet', extent=[-L/2, L/2, -L/2, L/2])
    plt.title("Near-Field Intensity")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(I_far, cmap='jet', extent=[-L/2, L/2, -L/2, L/2])
    plt.title(f"Intensity at z = {z:.2f} m")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
