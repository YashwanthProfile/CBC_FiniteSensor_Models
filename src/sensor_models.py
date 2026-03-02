import numpy as np
from src.nearfield_setupfuns import near_field
from src.beam_prop_funs import angular_spectrum

# ===================================================================
# Far-field support for sensor placement -> only to get support mask
# ===================================================================
def far_field_support(Nbeams,
                      X,Y,beam_centers,P,w0, 
                      Npix,dx,wavelength,k,z_prop, 
                      radius_energy=0.99):
    """
    Returns support mask for constraining the sensor locations in far-field
    
    :param Nbeams: Description
    :param radius_energy: Description
    """
    ph = np.random.uniform(0, 2*np.pi, Nbeams)
    I = np.abs(angular_spectrum(near_field(X,Y,ph,beam_centers,P,w0),  Npix, dx, wavelength, k, z_prop))**2
    flat = I.flatten()
    idx = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[idx]) / flat.sum()
    cutoff = flat[idx][np.searchsorted(cum, radius_energy)]
    return I >= cutoff

# ===================================================================
# Sensor templates (constrained)
# ===================================================================
def sensors_random(M, support_indices, target_pixel, sensor_at_targetpixel_flag=True):
    idx = np.random.choice(len(support_indices+1), M, replace=False)
    s = support_indices[idx][:, ::-1]

    if sensor_at_targetpixel_flag:
        s = np.vstack((s,target_pixel))
    
    return [tuple(p) for p in s]

def sensors_grid(M, support_mask, target_pixel):
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
        grid.append(target_pixel)

    return grid


def sensors_laser_matched(L,Npix,beam_centers,support_mask,target_pixel,sensor_at_targetpixel_flag=True):
    f = lambda u: int((u+L/2)/L*Npix)
    s = [(f(x), f(y)) for x, y in beam_centers]
    s = [p for p in s if support_mask[p[1], p[0]]]
    if sensor_at_targetpixel_flag:
        s = np.vstack((s,target_pixel))
    return s


# ===================================================================
# Gaussian Sensor
# ===================================================================
def gaussian_sensor(I, centers, sigma=6):
    """
    Docstring for gaussian_sensor
    
    :param I: Description
    :param centers: Description
    :param sigma: Description
    """
    Yg, Xg = np.indices(I.shape)
    y = np.zeros(len(centers))
    for i, (cx, cy) in enumerate(centers):
        w = np.exp(-((Xg-cx)**2 + (Yg-cy)**2)/(2*sigma**2))
        y[i] = np.sum(I*w)
    return y

# ===================================================================
# Bitmap Sensor
# ===================================================================
def bitmap_sensor(I, masks):
    return np.array([I[m].sum() for m in masks])