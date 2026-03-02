import numpy as np
from src.nearfield_setupfuns import near_field
from src.beam_prop_funs import angular_spectrum
from src.sensor_models import gaussian_sensor, bitmap_sensor

# =====================================================
# Power-in-Bucket (TRUE)
# =====================================================
def power_in_bucket(I, cx, cy, Npix, r=6):
    """
    Docstring for power_in_bucket
    
    :param I: Description
    :param cx: Description
    :param cy: Description
    :param Npix: Description
    :param r: Description
    """
    Yg, Xg = np.ogrid[:Npix, :Npix]
    mask = (Xg-cx)**2 + (Yg-cy)**2 <= r**2
    return I[mask].sum() / I.sum()

# =====================================================
# Merit functions
# =====================================================
def merit_full(phases, target,
               X,Y,beam_centers,P,w0,
               Npix,dx,wavelength,k,z_prop):
    """
    Docstring for merit_full
    
    :param phases: Description
    :param target: Description
    """
    I = np.abs(angular_spectrum(near_field(X,Y,phases,beam_centers,P,w0), 
                                Npix,dx,wavelength,k,z_prop))**2
    return power_in_bucket(I, *target, Npix)

def merit_gaussian(phases, centers, tidx,
                   X,Y,beam_centers,P,w0,
                   Npix,dx,wavelength,k,z_prop):
    """
    Docstring for merit_gaussian
    
    :param phases: Description
    :param centers: Description
    :param tidx: Description
    """
    I = np.abs(angular_spectrum(near_field(X,Y,phases,beam_centers,P,w0),
                                Npix,dx,wavelength,k,z_prop))**2
    y = gaussian_sensor(I, centers)
    return y[tidx] / y.sum()

def merit_bitmap(phases, masks, tidx):
    """
    Docstring for merit_bitmap
    
    :param phases: Description
    :param masks: Description
    :param tidx: Description
    """
    I = np.abs(angular_spectrum(near_field(phases)))**2
    y = bitmap_sensor(I, masks)
    return y[tidx] / y.sum()

# =====================================================
# Gradient 
# =====================================================
def gradient(phases, merit_fn, delta=1e-3):
    """
    Docstring for gradient
    
    :param phases: Description
    :param merit_fn: Description
    :param delta: Description
    """
    g = np.zeros_like(phases)
    J0 = merit_fn(phases)
    for i in range(len(phases)):
        dp = phases.copy()
        dp[i] += delta
        g[i] = (merit_fn(dp) - J0) / delta
    return g

# =====================================================
# Adagrad
# =====================================================
def adagrad(phases0, merit_fn, iters=150, alpha=0.2, eta=0.01):
    """
    Docstring for adagrad
    
    :param phases0: Description
    :param merit_fn: Description
    :param iters: Description
    :param alpha: Description
    :param eta: Description
    """
    phases = phases0.copy()
    G = np.zeros_like(phases)
    hist = []
    phase_hist = []

    for t in range(iters):
        g = gradient(phases, merit_fn)
        G += g**2
        alpha_t = (alpha/(1+t*eta))
        phases += alpha_t * g / (np.sqrt(G) + 1e-8)

        hist.append(merit_fn(phases))
        phase_hist.append(phases.copy())

        if t % 10 == 0:
            print(f"Iter {t:3d}/{iters:3d} | Relative-PIB = {hist[-1]:.4f}")

    return phases, np.array(hist), np.array(phase_hist)