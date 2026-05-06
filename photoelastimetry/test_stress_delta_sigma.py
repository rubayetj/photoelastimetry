import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button

import argparse
import os
import json5
import tifffile
import numpy as np
import photoelastimetry.calibrate
import photoelastimetry.io
import photoelastimetry.plotting
from photoelastimetry.main import _merge_params_with_calibration, _normalise_wavelengths
from photoelastimetry.seeding import phase_decomposed_seeding

# Re-use helpers from test_manual_calibration
from photoelastimetry.test_manual_calibration import (
    del_sigma_disk,
    DEL_SIGMA_ANALYTICAL,
    P_DISK, R_DISK, H_DISK, C_BOUNDS, N_POINTS,
    _normalise_input_stokes_vector,
    invert_wrapped_retardance,
    step2_only,
    _fit_circle_from_points,
    pick_circle_center,
    compute_del_sigma_predicted,
    print_stress_from_input_C,
    plot_rmse_vs_C,
    image_to_step2,
)


# ── Full pipeline: image → stress → delta_sigma ───────────────────────────────

def image_to_stress_delta_sigma(params, output_dir="."):
    """
    Run phase_decomposed_seeding to get delta_sigma directly (no optimiser),
    then extract delta_sigma at the disk center and compare with the analytical value.

    Parameters
    ----------
    params : dict
        JSON5 config (same as used for image_to_step2).
    output_dir : str
        Directory to save output TIFFs.

    Returns
    -------
    seed         : PhaseDecomposedSeed
    delta_sigma  : ndarray [H, W]      principal stress difference in Pa
    theta        : ndarray [H, W]      principal stress angle in radians
    ix, iy       : int                 disk center pixel coords
    """
    print("\n=== Running phase_decomposed_seeding ===")

    # ── Load & preprocess data ────────────────────────────────────────────────
    merged = _merge_params_with_calibration(params)
    if "folderName" in merged:
        data, metadata = photoelastimetry.io.load_raw(merged["folderName"])
    elif "input_filename" in merged:
        data, metadata = photoelastimetry.io.load_image(merged["input_filename"])
    else:
        raise ValueError("Either 'folderName' or 'input_filename' must be specified.")

    profile = merged.get("_calibration_profile")
    if profile is not None:
        data = photoelastimetry.calibrate.apply_blank_correction(data, profile["blank_correction"])

    if merged.get("crop") is not None:
        x1, x2, y1, y2 = merged["crop"]
        data = data[y1:y2, x1:x2]

    if merged.get("binning") is not None:
        binning = merged["binning"]
        data = photoelastimetry.io.bin_image(data, binning)

    # ── Extract seeding parameters from config ────────────────────────────────
    C = merged["C"]
    L = float(merged["thickness"])
    WAVELENGTHS = _normalise_wavelengths(merged["wavelengths"])
    NU = 1.0
    C_VALUES = np.asarray(C, dtype=float) if isinstance(C, (list, np.ndarray)) else np.full(len(WAVELENGTHS), float(C))
    S_I_HAT = np.array(merged["S_i_hat"], dtype=float)
    if len(S_I_HAT) == 2:
        S_I_HAT = np.append(S_I_HAT, 0.0)
    seeding_cfg = merged.get("seeding", {})
    n_max      = seeding_cfg.get("n_max", 6)
    sigma_max  = seeding_cfg.get("sigma_max", 10e6)
    correction_params = merged.get("correction", {})

    # ── Circle center: use value from JSON if present, else interactive ───────
    if "circle_center" in merged:
        cx, cy, r_px = merged["circle_center"]
        ix = int(round(cx)); iy = int(round(cy))
    else:
        cx, cy, ix, iy, r_px = pick_circle_center(data)

    # ── Run seeding ───────────────────────────────────────────────────────────
    seed = phase_decomposed_seeding(
        data, WAVELENGTHS, C_VALUES, NU, L,
        S_i_hat=S_I_HAT,
        sigma_max=sigma_max,
        n_max=n_max,
        correction_params=correction_params,
    )

    delta_sigma = seed.delta_sigma
    theta       = seed.theta

    # ── Extract delta_sigma at disk center — single pixel ─────────────────────
    ds_single = delta_sigma[iy, ix]

    # ── Extract delta_sigma at disk center — 3×3 patch average ───────────────
    r0, r1 = iy - 1, iy + 2
    c0, c1 = ix - 1, ix + 2
    ds_patch = delta_sigma[r0:r1, c0:c1].mean()

    print(f"\n── delta_sigma at disk center (col={ix}, row={iy}) ──────────────")
    print(f"  Δσ_analytical          = {DEL_SIGMA_ANALYTICAL / 1e3:.4f} kPa")
    print(f"  Δσ_seeding (single px) = {ds_single / 1e3:.4f} kPa  "
          f"  error = {(ds_single - DEL_SIGMA_ANALYTICAL) / 1e3:+.4f} kPa")
    print(f"  Δσ_seeding (3×3 patch) = {ds_patch / 1e3:.4f} kPa  "
          f"  error = {(ds_patch - DEL_SIGMA_ANALYTICAL) / 1e3:+.4f} kPa")
    print("─────────────────────────────────────────────────────────────────\n")

    # ── Plot delta_sigma map with disk center marked ───────────────────────────
    delta_sigma_kPa = delta_sigma / 1e3

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(delta_sigma_kPa, cmap="viridis")
    ax.scatter([ix], [iy], c="red", s=80, marker="+", linewidths=2, zorder=5,
               label=f"center ({ix},{iy})")
    fig.colorbar(im, ax=ax, label="Δσ (kPa)")
    ax.set_title("Principal stress difference  Δσ  from phase_decomposed_seeding")
    ax.legend(fontsize=9)
    plt.tight_layout()

    # ── Diameter scatter: analytical vs seeding Δσ ───────────────────────────
    scale_m_per_px = R_DISK / r_px

    W_img = delta_sigma.shape[1]
    col_min = max(0,     int(np.floor(cx - r_px)))
    col_max = min(W_img, int(np.ceil(cx  + r_px)) + 1)
    cols   = np.arange(col_min, col_max)
    x_phys = (cols - cx) * scale_m_per_px

    mask   = np.abs(x_phys) < R_DISK
    cols   = cols[mask]
    x_phys = x_phys[mask]

    ds_exp = delta_sigma[iy, cols] / 1e3
    ds_ana = np.array([del_sigma_disk(P_DISK, R_DISK, H_DISK, xp, 0.0) for xp in x_phys]) / 1e3
    x_norm = x_phys / R_DISK

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_norm, ds_exp, s=60, color="steelblue", alpha=0.7,
               label="Seeding (phase_decomposed_seeding)")
    ax.scatter(x_norm, ds_ana, s=20, color="tomato",    alpha=0.7,
               label="Analytical (Jaeger & Cook)")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Normalised position along diameter  (x / R)", fontsize=11)
    ax.set_ylabel("Δσ  (kPa)", fontsize=11)
    ax.set_title("Principal stress difference along horizontal diameter", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()  # show all figures at once

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    ds_path = os.path.join(output_dir, "delta_sigma.tiff")
    th_path = os.path.join(output_dir, "theta_solver.tiff")
    tifffile.imwrite(ds_path, delta_sigma_kPa.astype(np.float32))
    tifffile.imwrite(th_path, theta.astype(np.float32))
    print(f"Saved: {ds_path}")
    print(f"Saved: {th_path}")

    return seed, delta_sigma, theta, ix, iy


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run phase_decomposed_seeding and extract delta_sigma "
            "at the disk center, comparing with the analytical solution."
        )
    )
    parser.add_argument("json_filename", type=str, help="Path to the JSON5 parameter file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save output TIFFs (default: current directory).",
    )
    parser.add_argument(
        "--step2",
        action="store_true",
        help="Also run the step-2 only pipeline from test_manual_calibration.",
    )
    args = parser.parse_args()

    params = json5.load(open(args.json_filename, "r"))

    if args.step2:
        print("\n=== Running step-2 only pipeline ===")
        image_to_step2(params, output_dir=args.output_dir)

    image_to_stress_delta_sigma(params, output_dir=args.output_dir)