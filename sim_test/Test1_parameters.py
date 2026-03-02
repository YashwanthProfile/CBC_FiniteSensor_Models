import numpy as np

# =====================================================
# Physical & Simulation Parameters
# =====================================================
wavelength      = 1064e-9
k               = 2 * np.pi / wavelength
w0              = 5e-3
beam_spacing    = 12e-3
P               = 1.0
Npix            = 256
L               = 0.12
dx              = L / Npix
z_prop          = 100.0