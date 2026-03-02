import numpy as np
import matplotlib.pyplot as plt 
from datetime import date
import os

from src.nearfield_setupfuns import hexagonal_array, near_field
from src.beam_prop_funs import angular_spectrum
from src.sensor_models import far_field_support, sensors_grid, sensors_random, gaussian_sensor
from src.optimization_funs import power_in_bucket, merit_full, merit_gaussian, gradient, adagrad

# =====================================================
# Experiment metadata
# =====================================================
today = date.today().isoformat()
outdir = f"results/results_{today}"
os.makedirs(outdir, exist_ok=True)

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

RANDOM_SEED     = 0
ITERATIONS      = 200
FONTSIZE        = 15
LEGEND_FONTSIZE = int(0.8*FONTSIZE)
PLOT_SAVE_FLAG  = False
# =====================================================
# Spatial Grid
# =====================================================
x = np.linspace(-L/2, L/2, Npix)
y = np.linspace(-L/2, L/2, Npix)
X, Y = np.meshgrid(x, y)

# =====================================================
# Beam placement - near field
# =====================================================
beam_centers    = hexagonal_array(3, beam_spacing)
Nbeams          = len(beam_centers)
Msensors        = Nbeams*10
# =====================================================
# Far-field support for sensor placement
# =====================================================
support_mask = far_field_support(Nbeams,
                                 X,Y,beam_centers,P,w0,
                                 Npix,dx,wavelength,k,z_prop)

# =====================================================
# Sensor templates (constrained)
# =====================================================
target_pixel = (148, 148)
support_indices = np.column_stack(np.where(support_mask))


sensor_sets = {
    "Full Image": None,
    "Random": sensors_random(Msensors,support_indices,target_pixel),
    "Grid": sensors_grid(Msensors,support_mask,target_pixel),
}

# =====================================================
# Run experiments
# =====================================================
np.random.seed(RANDOM_SEED)
ph0 = np.random.uniform(0, 2*np.pi, Nbeams)
results = {}

for name, sensors in sensor_sets.items():
    print(f"\n--- {name} optimization ---")

    if name == "Full Image":
        ph, hist, ph_hist = adagrad(ph0, lambda p: merit_full(p, target_pixel, 
                                                              X,Y,beam_centers,P,w0,
                                                              Npix,dx,wavelength,k,z_prop),iters=ITERATIONS)
    else:
        ph, hist, ph_hist = adagrad(ph0, lambda p, s=sensors: merit_gaussian(p, s, -1,
                                                                             X,Y,beam_centers,P,w0,
                                                                             Npix,dx,wavelength,k,z_prop),iters=ITERATIONS)

    true_pib_hist = []
    for p in ph_hist:
        I = np.abs(angular_spectrum(near_field(X,Y,p,beam_centers,P,w0),
                                    Npix, dx, wavelength, k, z_prop))**2
        true_pib_hist.append(power_in_bucket(I, *target_pixel, Npix))

    I_final = np.abs(angular_spectrum(near_field(X,Y,ph,beam_centers,P,w0),
                                    Npix, dx, wavelength, k, z_prop))**2
    pib_final = power_in_bucket(I_final, *target_pixel, Npix)

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



# Collect all intensity fields first
near_I = np.abs(near_field(X, Y, ph0, beam_centers, P, w0))**2

far_Is = [r["I"] for r in results.values()]

# Global min and max for consistent colormap
all_I = [near_I] + far_Is
vmin = min(I.min() for I in all_I)
vmax = max(I.max() for I in all_I)



# =====================================================
# Figure A: Near-field + all Far-fields (shared color scale)
# =====================================================
plt.figure(figsize=(12,10))

# Near field
ax = plt.subplot(2,3,2)
im = ax.imshow(near_I, cmap='inferno', vmin=vmin, vmax=vmax)
ax.set_title("Near-field Intensity", fontsize=FONTSIZE)

i = 4
for name, r in results.items():
    ax = plt.subplot(2,3,i)
    ax.imshow(r["I"], cmap='inferno', vmin=vmin, vmax=vmax)
    ax.scatter(*target_pixel, c='cyan', marker='*', s=80)

    if r["sensors"] is not None:
        xs, ys = zip(*r["sensors"])
        ax.scatter(xs, ys, facecolors='none', edgecolors='white')

    ax.set_title(f"Far-field ({name})", fontsize=FONTSIZE)
    i += 1

# Single global colorbar
cbar = plt.colorbar(im, 
                    ax=plt.gcf().axes,
                    orientation="horizontal", 
                    fraction=0.025, 
                    pad=0.09)
cbar.set_label("Intensity (a.u.)", fontsize=FONTSIZE)

# plt.tight_layout()

if PLOT_SAVE_FLAG:
    plt.savefig(f"{outdir}/Fields_prop_{z_prop}_Msens_{Msensors}_{today}.png", bbox_inches="tight", dpi=200)
    plt.savefig(f"{outdir}/Fields_prop_{z_prop}_Msens_{Msensors}_{today}.pdf", bbox_inches="tight")
    print(f"Saved Fileds to: {outdir}")
else:
    plt.show()
# =====================================================
# Figure B: Sensor Objective Convergence
# =====================================================

plt.figure()
for name, r in results.items():
    plt.plot(r["obj_hist"], label=name)
plt.xlabel("Iteration")
plt.ylabel("Objective")
plt.legend(framealpha=1, edgecolor="k", fontsize=LEGEND_FONTSIZE)
plt.title(f"Sensor / PIB Objective Convergence: Propogated-{z_prop:.1f} (m)")
if PLOT_SAVE_FLAG:
    plt.savefig(f"{outdir}/True_PIB_Convergence_prop_{z_prop}_Msens_{Msensors}_{today}.png", dpi=200)
    plt.savefig(f"{outdir}/True_PIB_Convergence_prop_{z_prop}_Msens_{Msensors}_{today}.pdf", dpi=200)
    print(f"Saved Fileds to: {outdir}")
else:
    plt.show()
# =====================================================
# Figure C: TRUE PIB Convergence
# =====================================================
plt.figure()
for name, r in results.items():
    plt.plot(r["true_pib_hist"], label=name)
plt.xlabel("Iteration")
plt.ylabel("TRUE PIB")
plt.legend(framealpha=1, edgecolor="k", fontsize=LEGEND_FONTSIZE)
plt.title(f"True PIB Convergence: Propogated-{z_prop:.1f} (m)")
if PLOT_SAVE_FLAG:
    plt.savefig(f"{outdir}/True_PIB_Convergence_prop_{z_prop}_Msens_{Msensors}_{today}.png", dpi=200)
    plt.savefig(f"{outdir}/True_PIB_Convergence_prop_{z_prop}_Msens_{Msensors}_{today}.pdf", dpi=200)
    print(f"Saved Fileds to: {outdir}")
else:
    plt.show()
