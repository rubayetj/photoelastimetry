import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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

from photoelastimetry.test_manual_calibration import (
    step2_only,
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

    binning_factor = 1
    if merged.get("binning") is not None:
        binning_factor = int(merged["binning"])
        data = photoelastimetry.io.bin_image(data, binning_factor)

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

    # ── Upsample outputs to pre-binning resolution ────────────────────────────
    if binning_factor > 1:
        def _upsample(arr, f):
            return np.repeat(np.repeat(arr, f, axis=0), f, axis=1)
        delta_sigma_out = _upsample(delta_sigma, binning_factor)
        theta_out       = _upsample(theta,       binning_factor)
    else:
        delta_sigma_out = delta_sigma
        theta_out       = theta

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    ds_path = os.path.join(output_dir, "delta_sigma.tiff")
    th_path = os.path.join(output_dir, "theta_solver.tiff")
    tifffile.imwrite(ds_path, delta_sigma_out.astype(np.float32))
    tifffile.imwrite(th_path, theta_out.astype(np.float32))
    print(f"Saved: {ds_path}")
    print(f"Saved: {th_path}")

    # ── Load ground truth delta_sigma from stress TIFF ────────────────────────
    ds_true = None
    input_fn = merged.get("input_filename", "")
    stress_fn = input_fn.replace("_images.", "_stress.")
    if stress_fn != input_fn and os.path.exists(stress_fn):
        stress, _ = photoelastimetry.io.load_image(stress_fn)
        if merged.get("crop") is not None:
            stress = stress[y1:y2, x1:x2]
        sxx, syy, txy = stress[..., 0], stress[..., 1], stress[..., 2]
        ds_true = np.sqrt((sxx - syy) ** 2 + 4 * txy ** 2)
        print(f"Loaded ground truth: {stress_fn}")

    # ── Plot: full image side-by-side ─────────────────────────────────────────
    n_cols = 2 if ds_true is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    vmin = 0
    vmax = np.nanmax(delta_sigma_out) / 1e6
    if ds_true is not None:
        vmax = max(vmax, np.nanmax(ds_true) / 1e6)

    im0 = axes[0].imshow(delta_sigma_out / 1e6, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Seeded  Δσ")
    fig.colorbar(im0, ax=axes[0], label="Δσ (MPa)")

    if ds_true is not None:
        im1 = axes[1].imshow(ds_true / 1e6, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("Ground truth  Δσ")
        fig.colorbar(im1, ax=axes[1], label="Δσ (MPa)")

    plt.tight_layout()
    plt.show()

    # ── Plot: central row overlay ─────────────────────────────────────────────
    iy_out = delta_sigma_out.shape[0] // 2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(delta_sigma_out[iy_out, :] / 1e6, color="steelblue", lw=1.2, label="Seeded")
    if ds_true is not None:
        iy_true = ds_true.shape[0] // 2
        ax.plot(ds_true[iy_true, :] / 1e6, color="tomato", lw=1.2, ls="--", label="Ground truth")
    ax.set_xlabel("Column (px)", fontsize=11)
    ax.set_ylabel("Δσ  (MPa)", fontsize=11)
    ax.set_title(f"Central row (row {iy_out}) — Δσ across all columns", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return seed, delta_sigma, theta


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