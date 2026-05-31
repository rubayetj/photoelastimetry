import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from plot_config import configure_plots
from tqdm import tqdm

from photoelastimetry.generate.disk import generate_synthetic_brazil_test
from photoelastimetry.seeding import phase_decomposed_seeding

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


configure_plots()

# Photoelastic parameters
thickness = 0.01  # 10 mm
wavelengths = np.array([650e-9, 550e-9, 450e-9])  # R, G, B (meters)
C_values = np.array([4.0e-9, 4.0e-9, 4.0e-9]) * 10  # Stress-optic coeff (Pa^-1)
polarisation_efficiency = 1.0

# Parameters
nu_solid = 1.0
thickness = 0.01  # 10 mm
diameter = 0.1  # 100 mm
radius = diameter / 2
force = 2e-2  # Newtons
S_i_hat = np.array([0.0, 0.0, 1.0])  # Circular
n_max = 6
sigma_max = n_max * wavelengths.min() / (C_values.max() * nu_solid * thickness)
print(f"Max stress for seeding: {sigma_max:.2e} Pa")

n_trials = 2
snr = 1000000.0

# Define grid
resolution = 32
x = np.linspace(-radius * 1.2, radius * 1.2, resolution)
y = np.linspace(-radius * 1.2, radius * 1.2, resolution)
X, Y = np.meshgrid(x, y)
mask = X**2 + Y**2 <= radius**2

cb_fraction = 0.05
cb_pad = 0.02
cb_shrink = 0.8

print(f"Generating synthetic disk (F={force}N, R={radius}m)...")
# Get ground truth stresses
# synthetic_images shape: (H, W, 3, 4)
synthetic_images, pd_true, theta_true, _, _, _ = generate_synthetic_brazil_test(
    X, Y, force / thickness, radius, S_i_hat, mask, wavelengths, thickness, C_values, 1.0
)

# Fix NaNs in synthetic images for Poisson generation
synthetic_images[~mask] = 0.0
synthetic_images = np.nan_to_num(synthetic_images, nan=0.0)
# Ensure non-negative
synthetic_images = np.maximum(synthetic_images, 0.0)

# Storage for error stats
mae_grad_history = []
med_rel_grad_history = []
mae_theta_history = []
mae_theta_robust_history = []
delta_sigma_rec_history = []
theta_rec_history = []

print(f"Running {n_trials} Monte Carlo trials with SNR={snr}...")

for _ in tqdm(range(n_trials)):
    # 1. Add Noise
    n_photons_max = snr**2
    if n_photons_max > 1e-6:
        noisy_images = np.random.poisson(synthetic_images * n_photons_max) / n_photons_max
    else:
        noisy_images = synthetic_images.copy()

    # 2. Seeding / Inversion
    seed = phase_decomposed_seeding(
        noisy_images,
        wavelengths,
        C_values,
        nu=nu_solid,
        L=thickness,
        S_i_hat=S_i_hat,
        sigma_max=sigma_max,
        n_max=n_max,
        # correction_params={"unwrap_angles": True},
    )

    delta_sigma_est = seed.delta_sigma
    theta_est = seed.theta

    # 3. Error Calculation (Inside mask)
    err_pd = np.abs(delta_sigma_est - pd_true)
    rel_err_pd = err_pd / np.maximum(pd_true, 1e-12)

    # Angle Error
    diff_theta = np.abs(theta_est - theta_true)
    diff_theta = np.minimum(diff_theta, np.pi - diff_theta)
    robust_mask = mask & (pd_true > 0.1 * sigma_max)

    mae_grad = np.nanmean(err_pd[mask])
    med_rel_grad = np.nanmedian(rel_err_pd[mask])
    mae_theta = np.nanmean(diff_theta[mask])
    mae_theta_robust = np.nanmean(diff_theta[robust_mask]) if np.any(robust_mask) else np.nan

    mae_grad_history.append(mae_grad)
    med_rel_grad_history.append(med_rel_grad)
    mae_theta_history.append(mae_theta)
    mae_theta_robust_history.append(mae_theta_robust)
    delta_sigma_rec_history.append(delta_sigma_est)
    theta_rec_history.append(theta_est)

# Stats
mae_grad_mean = np.mean(mae_grad_history)
mae_grad_std = np.std(mae_grad_history)
med_rel_grad_mean = np.mean(med_rel_grad_history)
med_rel_grad_std = np.std(med_rel_grad_history)
mae_theta_mean = np.mean(mae_theta_history)
mae_theta_std = np.std(mae_theta_history)
mae_theta_robust_mean = np.nanmean(mae_theta_robust_history)
mae_theta_robust_std = np.nanstd(mae_theta_robust_history)

# print(f"MAE Stress Delta: {mae_grad_mean:.2e} +/- {mae_grad_std:.2e} Pa")
# print(f"Median Relative Stress Error: {med_rel_grad_mean:.2e} +/- {med_rel_grad_std:.2e}")
# print(f"MAE Theta: {mae_theta_mean:.2e} +/- {mae_theta_std:.2e} rad")
# print(
#     f"MAE Theta (DeltaSigma > 0.1*DeltaSigma_max): {mae_theta_robust_mean:.2e} +/- {mae_theta_robust_std:.2e} rad"
# )

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(6.04, 3.0))

# Median reconstructions across trials for plotting
ds_rec = np.nanmedian(np.stack(delta_sigma_rec_history, axis=0), axis=0)

theta_stack = np.stack(theta_rec_history, axis=0)
sin2theta_med = np.nanmedian(np.sin(2 * theta_stack), axis=0)
cos2theta_med = np.nanmedian(np.cos(2 * theta_stack), axis=0)
theta_rec = 0.5 * np.arctan2(sin2theta_med, cos2theta_med)

# Mask reconstructions for plotting
ds_rec = ds_rec.copy()
theta_rec = theta_rec.copy()
ds_rec[~mask] = np.nan
theta_rec[~mask] = np.nan

# --- Magnitudes (Stress Difference) ---
# True
im0 = axes[0, 0].imshow(pd_true / sigma_max, cmap="inferno", origin="lower")
# axes[0, 0].set_title(r"$\Delta\sigma / \Delta\sigma_\mathrm{max}$ (Analytical)")
axes[0, 0].text(0, 1.0, "(a)", transform=axes[0, 0].transAxes)
axes[0, 0].axis("off")
cb = plt.colorbar(im0, ax=axes[0, 0], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink, ticks=[0, 0.5, 1.0])
cb.set_label(r"$\Delta\sigma/\Delta\sigma_\mathrm{max}$")

# Recovered
im1 = axes[0, 1].imshow(ds_rec / sigma_max, cmap="inferno", vmin=0, vmax=1, origin="lower")
# axes[0, 1].set_title(r"$\Delta\sigma / \Delta\sigma_\mathrm{max}$ (Recovered)")
axes[0, 1].text(0, 1.0, "(b)", transform=axes[0, 1].transAxes)
axes[0, 1].axis("off")
cb = plt.colorbar(im1, ax=axes[0, 1], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink, ticks=[0, 0.5, 1.0])
cb.set_label(r"$\Delta\sigma/\Delta\sigma_\mathrm{max}$")

# Error
err_pd_last = np.abs((ds_rec - pd_true) / sigma_max)
im2 = axes[0, 2].imshow(err_pd_last, cmap="inferno", origin="lower", vmin=0, vmax=1)  # Cap error scale
# axes[0, 2].set_title(r"$\Delta\sigma / \Delta\sigma_\mathrm{max}$ Error")
axes[0, 2].text(0, 1.0, "(c)", transform=axes[0, 2].transAxes)
axes[0, 2].axis("off")
cb = plt.colorbar(im2, ax=axes[0, 2], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink, ticks=[0, 0.5, 1.0])
# cb.ax.set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])
cb.set_label(r"$\Delta\sigma/\Delta\sigma_\mathrm{max}$")

# --- Angles (Theta) ---
# Use hsv or phase cmap
cmap_phase = "twilight"
# True
im3 = axes[1, 0].imshow(theta_true, cmap=cmap_phase, origin="lower", vmin=-np.pi / 2, vmax=np.pi / 2)
# axes[1, 0].set_title(r"$\theta$ (Analytical)")
axes[1, 0].text(0, 1.0, "(d)", transform=axes[1, 0].transAxes)
axes[1, 0].axis("off")
cb = plt.colorbar(
    im3, ax=axes[1, 0], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink, ticks=[-np.pi / 2, 0, np.pi / 2]
)
cb.ax.set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])
cb.set_label(r"$\theta$ (rad)", labelpad=-5)

# Recovered
im4 = axes[1, 1].imshow(theta_rec, cmap=cmap_phase, origin="lower", vmin=-np.pi / 2, vmax=np.pi / 2)
# axes[1, 1].set_title(r"$\theta$ (Recovered)")
axes[1, 1].text(0, 1.0, "(e)", transform=axes[1, 1].transAxes)
axes[1, 1].axis("off")
cb = plt.colorbar(
    im4, ax=axes[1, 1], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink, ticks=[-np.pi / 2, 0, np.pi / 2]
)
cb.ax.set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])
cb.set_label(r"$\theta$ (rad)", labelpad=-5)

# Error
diff_theta_last = np.abs(theta_rec - theta_true)
diff_theta_last = np.minimum(diff_theta_last, np.pi - diff_theta_last)
im5 = axes[1, 2].imshow(
    diff_theta_last, cmap="inferno", origin="lower", vmin=0, vmax=np.pi / 2
)  # Cap error scale
# axes[1, 2].set_title(r"$\theta$ Error")
axes[1, 2].text(0, 1.0, "(f)", transform=axes[1, 2].transAxes)
axes[1, 2].axis("off")
cb = plt.colorbar(
    im5,
    ax=axes[1, 2],
    fraction=cb_fraction,
    pad=cb_pad,
    shrink=cb_shrink,
    ticks=[0, np.pi / 4, np.pi / 2],
)
cb.ax.set_yticklabels(["0", r"$\pi/8$", r"$\pi/4$"])
cb.set_label(r"$\theta$ (rad)")

plt.subplots_adjust(left=0.0, bottom=0.02, right=0.91, top=0.96, hspace=0.2, wspace=0.2)

# Save
output_path = "papers/validate_disk.pdf"
plt.savefig(output_path, dpi=300)
print(f"Saved validation figure to {output_path}")

try:
    shutil.copy(
        output_path,
        os.path.expanduser("~/Dropbox/Apps/Overleaf/RGB Granular Photoelasticity/images/validate_disk.pdf"),
    )
    print("Copied to Dropbox")
except Exception as e:
    print(f"Could not copy to dropbox: {e}")