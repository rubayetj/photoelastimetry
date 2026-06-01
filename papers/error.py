"""
error.py — Pixel-wise error between experimental Δσ (phase_decomposed_seeding)
           and the analytical Brazil-test solution.

Usage:
    python papers/error.py json/disk.json5 [--output_dir .]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.widgets import Button

# Discrete error-band colormap
_BAND_BOUNDS = [0, 10, 25, 30, 60, 80, 200]   # 200 acts as ∞
_BAND_COLORS = ["#4caf50", "#ffeb3b", "#ff9800", "#f44336", "#9c27b0", "#212121"]
_BAND_LABELS = ["<10 %", "10–25 %", "25–30 %", "30–60 %", "60–80 %", ">80 %"]
_BAND_CMAP   = mcolors.ListedColormap(_BAND_COLORS, name="error_bands")
_BAND_NORM   = mcolors.BoundaryNorm(_BAND_BOUNDS, ncolors=len(_BAND_COLORS))
import json5
import numpy as np
import tifffile

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import photoelastimetry.calibrate
import photoelastimetry.io
from photoelastimetry.generate.disk import pointload_stress, diametrical_stress_cartesian, huang2014
from photoelastimetry.main import _merge_params_with_calibration, _normalise_wavelengths
from photoelastimetry.seeding import phase_decomposed_seeding

# -- Disk constants (Disk 2) ---------------------------------------------------
Load   = 600              # grams

R_DISK = 14.54/(1000*2)         # m  (radius) DISK 1
H_DISK = 0.00699                # m  (thickness) DISK 1
#R_DISK = 17.41/(1000*2)         # m  (radius) DISK 2
#H_DISK = 0.00686                # m  (thickness) DISK 2
P_DISK = Load * 9.81 / (1000)    # N

# -- Circle picker -------------------------------------------------------------

def _fit_circle_from_points(points):
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x**2 + y**2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = result
    radius = np.sqrt(c + cx**2 + cy**2)
    if radius <= 0:
        raise ValueError("Circle fit produced a non-positive radius.")
    return cx, cy, radius


def pick_circle_center(data):
    if data.ndim == 4:
        preview = data[:, :, 0, :].mean(axis=-1).astype(float)
    else:
        preview = data[:, :, 0].astype(float)
    preview = (preview - preview.min()) / (preview.max() - preview.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(bottom=0.16)
    ax.imshow(preview, cmap="gray")
    ax.set_title("Circle Center Picker — left-click circumference, right-click to undo")
    ax.set_axis_off()

    points = []
    point_scatter  = ax.scatter([], [], c="yellow", s=24, zorder=4)
    center_scatter = ax.scatter([], [], c="red", s=80, marker="+", linewidths=2, zorder=5)
    circle_patch   = Circle((0, 0), radius=1, fill=False, edgecolor="cyan",
                             linewidth=2, visible=False)
    ax.add_patch(circle_patch)

    status_text = ax.text(
        0.01, 0.99,
        "Left-click circumference points (>=3). Right-click to undo.",
        transform=ax.transAxes, va="top", ha="left", fontsize=9, color="white",
        bbox={"facecolor": "black", "alpha": 0.5, "pad": 4},
    )

    state = {"accepted": False, "fit": None}

    def _update():
        if points:
            point_scatter.set_offsets(np.array(points))
        else:
            point_scatter.set_offsets(np.empty((0, 2)))
        if len(points) >= 3:
            try:
                cx, cy, r = _fit_circle_from_points(points)
                circle_patch.center = (cx, cy)
                circle_patch.radius = r
                circle_patch.set_visible(True)
                center_scatter.set_offsets([[cx, cy]])
                status_text.set_text(
                    f"n={len(points)}  center=({cx:.1f},{cy:.1f})  "
                    f"radius={r:.1f}px  <- Done to accept"
                )
                state["fit"] = (cx, cy, r)
            except ValueError as e:
                circle_patch.set_visible(False)
                center_scatter.set_offsets(np.empty((0, 2)))
                status_text.set_text(str(e))
                state["fit"] = None
        else:
            circle_patch.set_visible(False)
            center_scatter.set_offsets(np.empty((0, 2)))
            status_text.set_text(f"{len(points)}/3 points — keep clicking the circumference.")
            state["fit"] = None
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button == 1:
            points.append((float(event.xdata), float(event.ydata)))
        elif event.button == 3 and points:
            points.pop()
        _update()

    def _on_done(_):
        if state["fit"] is None:
            status_text.set_text("Need a valid circle fit before clicking Done.")
            fig.canvas.draw_idle()
            return
        state["accepted"] = True
        plt.close(fig)

    def _on_reset(_):
        points.clear()
        _update()

    done_ax  = fig.add_axes([0.80, 0.03, 0.16, 0.07])
    reset_ax = fig.add_axes([0.62, 0.03, 0.16, 0.07])
    done_btn  = Button(done_ax,  "Done")
    reset_btn = Button(reset_ax, "Reset")
    done_btn.on_clicked(_on_done)
    reset_btn.on_clicked(_on_reset)
    fig.canvas.mpl_connect("button_press_event", _on_click)
    _update()
    plt.show()

    if not state["accepted"] or state["fit"] is None:
        raise ValueError("Interactive selection cancelled — circle was not accepted.")

    cx, cy, r_px = state["fit"]
    H, W = data.shape[:2]
    ix = int(np.clip(round(cx), 0, W - 1))
    iy = int(np.clip(round(cy), 0, H - 1))
    print(f"Circle center -> subpixel: ({cx:.2f}, {cy:.2f})  pixel: col={ix}, row={iy}  r={r_px:.1f} px")
    return cx, cy, ix, iy, r_px


# -- Load experimental stress --------------------------------------------------

def load_experimental_stress(params):
    """
    Load image data described by JSON5 params, run phase_decomposed_seeding,
    and return the experimental Δσ map and disk geometry.

    Returns
    -------
    delta_sigma : ndarray [H, W]   principal stress difference (Pa)
    theta       : ndarray [H, W]   principal stress angle (rad)
    cx, cy      : float            subpixel disk centre (pixels)
    ix, iy      : int              integer disk centre (pixels)
    r_px        : float            disk radius (pixels)
    """
    merged = _merge_params_with_calibration(params)

    if "folderName" in merged:
        data, _ = photoelastimetry.io.load_raw(merged["folderName"])
    elif "input_filename" in merged:
        data, _ = photoelastimetry.io.load_image(merged["input_filename"])
    else:
        raise ValueError("Either 'folderName' or 'input_filename' must be specified.")

    profile = merged.get("_calibration_profile")
    if profile is not None:
        data = photoelastimetry.calibrate.apply_blank_correction(
            data, profile["blank_correction"]
        )

    if merged.get("crop") is not None:
        x1, x2, y1, y2 = merged["crop"]
        data = data[y1:y2, x1:x2]

    if merged.get("binning") is not None:
        binning = merged["binning"]
        data = photoelastimetry.io.bin_image(data, binning)

    C           = merged["C"]
    L           = float(merged["thickness"])
    WAVELENGTHS = _normalise_wavelengths(merged["wavelengths"])
    C_VALUES    = (np.asarray(C, dtype=float)
                   if isinstance(C, (list, np.ndarray))
                   else np.full(len(WAVELENGTHS), float(C)))
    S_I_HAT     = np.array(merged["S_i_hat"], dtype=float)
    if len(S_I_HAT) == 2:
        S_I_HAT = np.append(S_I_HAT, 0.0)

    seeding_cfg       = merged.get("seeding", {})
    n_max             = seeding_cfg.get("n_max", 6)
    sigma_max         = seeding_cfg.get("sigma_max", 10e6)
    correction_params = merged.get("correction", {})

    if "circle_center" in merged:
        cx, cy, r_px = merged["circle_center"]
        ix = int(round(cx))
        iy = int(round(cy))
    else:
        cx, cy, ix, iy, r_px = pick_circle_center(data)

    print("\n=== Running phase_decomposed_seeding ===")
    seed = phase_decomposed_seeding(
        data, WAVELENGTHS, C_VALUES, 1.0, L,
        S_i_hat=S_I_HAT,
        sigma_max=sigma_max,
        n_max=n_max,
        correction_params=correction_params,
    )

    H, W = seed.delta_sigma.shape
    print(f"Experimental stress image size: {W} x {H}  (W x H)")
    return seed.delta_sigma, seed.theta, cx, cy, ix, iy, r_px


# -- Analytical stress field ---------------------------------------------------

def analytical_stress_field(H, W, cx, cy, r_px):
    """
    Compute analytical Δσ at every pixel using the pointload (K.Ramesh)
    formula scaled to the experimental pixel grid.

    Returns
    -------
    ds_ana    : ndarray [H, W]  analytical Δσ (Pa)
    disk_mask : ndarray [H, W]  True inside disk boundary
    """
    scale = R_DISK / r_px   # metres per pixel

    col_idx, row_idx = np.meshgrid(np.arange(W), np.arange(H))
    x_phys = (col_idx - cx) * scale
    y_phys = (row_idx - cy) * scale

    sx, sy, txy = huang2014(x_phys, y_phys, P_DISK, R_DISK, H_DISK, 15)
    #sx, sy, txy = diametrical_stress_cartesian(x_phys, y_phys, P_DISK / H_DISK, R_DISK)
    ds_ana = np.sqrt((sx - sy)**2 + 4.0 * txy**2)

    disk_mask = (x_phys**2 + y_phys**2) <= R_DISK**2
    return ds_ana, disk_mask


# -- Error metrics -------------------------------------------------------------

def stress_error(ds_exp, ds_ana, disk_mask):
    """
    Pixel-wise absolute and relative percentage errors inside the disk.

    Relative error is normalised by the local analytical value at each pixel:
        err_pct = |Δσ_exp - Δσ_ana| * 100 / max(Δσ_ana, 1e-12)

    Returns
    -------
    err_abs : ndarray [H, W]  |Δσ_exp - Δσ_ana| (Pa),              NaN outside disk
    err_pct : ndarray [H, W]  % error relative to local analytical, NaN outside disk
    mae     : float           mean absolute error inside disk (Pa)
    mre_pct : float           median % error inside disk
    """
    err_abs = np.abs(ds_exp - ds_ana)
    err_pct = err_abs * 100.0 / np.maximum(ds_ana, 1e-12)

    mae     = np.nanmean(err_abs[disk_mask])
    mre_pct = np.nanmedian(err_pct[disk_mask])

    err_abs_out = np.full_like(err_abs, np.nan)
    err_pct_out = np.full_like(err_pct, np.nan)
    err_abs_out[disk_mask] = err_abs[disk_mask]
    err_pct_out[disk_mask] = err_pct[disk_mask]

    return err_abs_out, err_pct_out, mae, mre_pct


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute pixel-wise error between experimental and analytical Δσ."
    )
    parser.add_argument("json_filename", type=str, help="Path to JSON5 params file.")
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Directory to save outputs (default: current directory).",
    )
    args = parser.parse_args()

    params = json5.load(open(args.json_filename, "r"))

    # Step 1: get experimental stress from real image
    ds_exp, theta, cx, cy, ix, iy, r_px = load_experimental_stress(params)

    # Step 2: generate analytical stress at same image size and disk center
    H, W = ds_exp.shape
    ds_ana, disk_mask = analytical_stress_field(H, W, cx, cy, r_px)

    # Mask both fields to disk boundary for display
    ds_exp_plot = ds_exp.copy()
    ds_exp_plot[~disk_mask] = np.nan

    ds_ana_plot = ds_ana.copy()
    ds_ana_plot[~disk_mask] = np.nan

    # Step 3: compute pixel-wise error
    ds_ana_center = ds_ana[iy, ix]
    ds_exp_center = ds_exp[iy, ix]
    err_abs, err_pct, mae, mre_pct = stress_error(ds_exp, ds_ana, disk_mask)

    err_abs_center = err_abs[iy, ix]
    err_pct_center = err_pct[iy, ix]

    print(f"\n-- Stress prediction error ---------------------------------------")
    print(f"  Image size           : {W} x {H}  px")
    print(f"  Disk center          : col={ix}, row={iy}   r={r_px:.1f} px")
    print(f"  Δσ experimental      = {ds_exp_center / 1e3:.4f} Pa")
    print(f"  Δσ analytical        = {ds_ana_center / 1e3:.4f} Pa")
    print(f"  Error at center      = {err_abs_center / 1e3:.4f} Pa  ({err_pct_center:.2f} %)")
    print(f"  MAE (whole disk)     = {mae / 1e3:.4f} Pa")
    print(f"  MRE (whole disk)     = {mre_pct:.2f} %")
    print("-----------------------------------------------------------------\n")

    # -- Diameter error profile ------------------------------------------------
    scale = R_DISK / r_px
    col_min = max(0,     int(np.floor(cx - r_px)))
    col_max = min(W,     int(np.ceil(cx  + r_px)) + 1)
    cols    = np.arange(col_min, col_max)
    x_phys  = (cols - cx) * scale
    in_disk = np.abs(x_phys) < R_DISK
    cols    = cols[in_disk]
    x_phys  = x_phys[in_disk]
    x_norm  = x_phys / R_DISK

    print(f"-- Diameter error profile (row={iy}) ----------------------------")
    print(f"  {'row':>4}  {'col':>4}  {'x/R':>7}  {'Exp (kPa)':>11}  {'Ana (kPa)':>11}  {'|Err|/Ana (%)':>14}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*11}  {'-'*11}  {'-'*14}")

    y_diam = (iy - cy) * scale   # subpixel y-offset of the diameter row
    sx, sy, txy = pointload_stress(
        x_phys[np.newaxis, :],
        np.full((1, len(x_phys)), y_diam),
        P_DISK, R_DISK, H_DISK,
    )
    ana_diam = np.sqrt((sx - sy)**2 + 4.0 * txy**2).ravel()

    for c, xn, exp_v, ana_v in zip(cols, x_norm, ds_exp[iy, cols], ana_diam):
        rel = abs(exp_v - ana_v) / max(ana_v, 1e-12) * 100
        print(f"  {iy:>4}  {c:>4}  {xn:>7.3f}  {exp_v/1e3:>11.4f}  {ana_v/1e3:>11.4f}  {rel:>14.2f}")
    print("-----------------------------------------------------------------\n")

    # Step 4: save error TIFFs
    os.makedirs(args.output_dir, exist_ok=True)
    abs_path = os.path.join(args.output_dir, "stress_error_abs.tiff")
    pct_path = os.path.join(args.output_dir, "stress_error_pct.tiff")
    tifffile.imwrite(abs_path, err_abs.astype(np.float32))
    tifffile.imwrite(pct_path, err_pct.astype(np.float32))
    print(f"Saved: {abs_path}")
    print(f"Saved: {pct_path}")

    csv_path = os.path.join(args.output_dir, "rel_error_matrix.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    np.savetxt(csv_path, err_pct, delimiter=",", fmt="%.4f")
    print(f"Saved: {csv_path}  ({H}x{W}, values in %, NaN outside disk)")

    # Step 5: plot experimental | analytical | % error
    vmin_stress = 0
    vmax_stress = np.nanpercentile(
        np.concatenate([ds_exp_plot[disk_mask], ds_ana_plot[disk_mask]]), 98
    ) / 1e3
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Stress prediction error   MAE = {mae/1e3:.2f} kPa   MRE = {mre_pct:.1f} %",
        fontsize=13,
    )

    im0 = axes[0].imshow(ds_exp_plot / 1e3, cmap="viridis_r", origin="upper",
                         vmin=vmin_stress, vmax=vmax_stress)
    axes[0].set_title("Experimental  $\\Delta\\sigma$  (Pa)")
    axes[0].scatter([ix], [iy], c="red", s=60, marker="+", linewidths=2, zorder=5)
    fig.colorbar(im0, ax=axes[0], label="Pa")
    axes[0].axis("off")

    im1 = axes[1].imshow(ds_ana_plot / 1e3, cmap="viridis_r", origin="upper",
                         vmin=vmin_stress, vmax=vmax_stress)
    axes[1].set_title("Analytical  $\\Delta\\sigma$  (Pa)")
    axes[1].scatter([ix], [iy], c="red", s=60, marker="+", linewidths=2, zorder=5)
    fig.colorbar(im1, ax=axes[1], label="Pa")
    axes[1].axis("off")

    im2 = axes[2].imshow(err_pct, cmap=_BAND_CMAP, norm=_BAND_NORM, origin="upper")
    axes[2].set_title("Relative error  (%)")
    cb2 = fig.colorbar(im2, ax=axes[2], ticks=_BAND_BOUNDS[:-1])
    cb2.ax.set_yticklabels(_BAND_LABELS)
    axes[2].axis("off")

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "stress_error.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")

    # -- Raw relative error (continuous colormap) --------------------------------
    fig_raw, ax_raw = plt.subplots(figsize=(6, 5))
    vmax_raw = np.nanpercentile(err_pct[disk_mask], 98)
    im_raw = ax_raw.imshow(err_pct, cmap="inferno", origin="upper", vmin=0, vmax=vmax_raw)
    fig_raw.colorbar(im_raw, ax=ax_raw, label="Relative error (%)")
    ax_raw.set_title(f"Raw relative error  (MRE = {mre_pct:.1f} %)")
    ax_raw.axis("off")
    plt.tight_layout()
    raw_path = os.path.join(args.output_dir, "stress_error_raw.png")
    plt.savefig(raw_path, dpi=150)
    print(f"Saved: {raw_path}")

    # -- Pixel-level error map ---------------------------------------------------
    px_scale = 20                        # display pixels per data pixel
    fig_h = err_pct.shape[0] * px_scale / 100
    fig_w = err_pct.shape[1] * px_scale / 100

    fig2, ax2 = plt.subplots(figsize=(fig_w + 1.5, fig_h))
    im_px = ax2.imshow(
        err_pct, cmap=_BAND_CMAP, norm=_BAND_NORM, origin="upper",
        interpolation="nearest",
        aspect="equal",
    )
    # pixel grid
    ax2.set_xticks(np.arange(-0.5, err_pct.shape[1], 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, err_pct.shape[0], 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=0.3, alpha=0.4)
    ax2.tick_params(which="minor", length=0)
    ax2.set_xticks(np.arange(0, err_pct.shape[1], 5))
    ax2.set_yticks(np.arange(0, err_pct.shape[0], 5))
    ax2.scatter([ix], [iy], c="cyan", s=60, marker="+", linewidths=1.5, zorder=5)
    cb_px = fig2.colorbar(im_px, ax=ax2, ticks=_BAND_BOUNDS[:-1], shrink=0.8)
    cb_px.ax.set_yticklabels(_BAND_LABELS)
    ax2.set_title("Relative error per pixel  (% of local analytical Δσ)")
    plt.tight_layout()
    pxmap_path = os.path.join(args.output_dir, "stress_error_pct.png")
    plt.savefig(pxmap_path, dpi=100)
    print(f"Saved: {pxmap_path}")

    plt.show()
