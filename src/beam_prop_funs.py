import numpy as np

def angular_spectrum(E0, Npix, dx, wavelength, k, z_prop):
    """
    Docstring for angular_spectrum
    
    :param E: Description
    :param Npix: Description
    :param dx: Description
    :param wavelength: Description
    :param k: Description
    :param z_prop: Description
    """
    fx = np.fft.fftfreq(Npix, dx)
    fy = np.fft.fftfreq(Npix, dx)
    FX, FY = np.meshgrid(fx, fy)
    arg = 1 - (wavelength*FX)**2 - (wavelength*FY)**2
    arg[arg < 0] = 0
    H = np.exp(1j * k * z_prop * np.sqrt(arg))
    return np.fft.ifft2(np.fft.fft2(E0) * H)