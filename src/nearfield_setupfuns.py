import numpy as np


def hexagonal_array(n_rings, spacing):
    """
    Returns co-ordinates for the beam centers
    
    :param n_rings: Number of hexagonal rings 
    :param spacing: -
    """
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

def near_field(X, Y, phases, beam_centers, P, w0):
    """
    Returns near-field intensity distribution (E0)
    
    :param X: Meshgrid X
    :param Y: Meshgrid Y
    :param phases: phase distribution for lasers (rad)
    :param beam_centers: location of laser centers in (x,y) (m)
    :param P: Power per beam (a.u.)
    :param w0: Beam wasit at near-field (m)
    
    """
    E0 = np.zeros_like(X, dtype=complex)
    for (xn, yn), phi in zip(beam_centers, phases):
        r2 = (X-xn)**2 + (Y-yn)**2
        E0 += np.sqrt(2*P/(np.pi*w0**2)) * np.exp(-r2/w0**2) * np.exp(-1j*phi)
    return E0