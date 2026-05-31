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
from photoelastimetry.image import compute_normalised_stokes, compute_stokes_components
from photoelastimetry.main import _merge_params_with_calibration, _normalise_wavelengths
from photoelastimetry.generate.disk import diametrical_stress_cartesian, huang2014



#──Constants─────────────────────────────────────────────────────────────────

# Disk geometry & load (hardcoded defaults)
Load =  600         #grams
P_DISK = Load*9.81/1000         # N
R_DISK = 14.54/(1000*2)         # m  (radius) DISK 1
H_DISK = 0.00699                # m  (thickness) DISK 1
#R_DISK = 17.41/(1000*2)         # m  (radius) DISK 2
#H_DISK = 0.00686                # m  (thickness) DISK 2
C_BOUNDS = (1e-15, 1e-4)
N_POINTS = 5000



def del_sigma_disk(P, R, h, x, y):
    """
    Principal stress difference √[(σ_x−σ_y)²+4τ_xy²] for a diametrically
    loaded disk at point (x, y). Uses the Jaeger & Cook exact solution.

    Parameters
    ----------
    P : float   Total load (N)
    R : float   Radius (m)
    h : float   Thickness (m)
    x : float   x-coordinate from disk centre (m)
    y : float   y-coordinate from disk centre (m)

    Returns
    -------
    float  √[(σ_x−σ_y)²+4τ_xy²]  (Pa)
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    Y = np.atleast_2d(np.asarray(y, dtype=float))
    sx, sy, txy = huang2014(X, Y, P, R, h, 15)
    return float(np.squeeze(np.sqrt((sx - sy)**2 + 4.0 * txy**2)))
    

# Evaluate at disk centre (x=0, y=0) with hardcoded values
DEL_SIGMA_ANALYTICAL = del_sigma_disk(P_DISK, R_DISK, H_DISK, x= 0, y= 0)


# ── Step 2 helpers ────────────────────────────────────────────────────────────

def _normalise_input_stokes_vector(S_i_hat):
    S_i_hat = np.asarray(S_i_hat, dtype=float)
    if S_i_hat.shape == (2,):
        S_i_hat = np.append(S_i_hat, 0.0)
    if S_i_hat.shape != (3,):
        raise ValueError(f"S_i_hat must have shape (2,) or (3,), got {S_i_hat.shape}")
    return S_i_hat


def invert_wrapped_retardance(S_m_hat, S_i_hat):
    s1 = S_m_hat[..., 0]
    s2 = S_m_hat[..., 1]

    is_circular = abs(S_i_hat[2]) > 0.9

    if is_circular:
        S3_in = S_i_hat[2]
        x_sum = np.sum(-s2 * S3_in, axis=-1)
        y_sum = np.sum(s1 * S3_in, axis=-1)
        theta = 0.5 * np.arctan2(y_sum, x_sum)
        magnitude = np.sqrt(s1**2 + s2**2)
        magnitude = np.clip(magnitude, 0, 1)
        delta_wrap = np.arcsin(magnitude)
        return theta, delta_wrap

    S1_in = S_i_hat[0]
    S2_in = S_i_hat[1]
    alpha = 0.5 * np.arctan2(S2_in, S1_in)
    dx = s2 - S2_in
    dy = S1_in - s1
    x_sum = np.sum(dx, axis=-1)
    y_sum = np.sum(dy, axis=-1)
    theta_mean = 0.5 * np.arctan2(y_sum, x_sum)
    sin_weight = np.abs(np.sin(2 * (theta_mean - alpha)))
    R = np.sqrt(dx**2 + dy**2)
    raw_magnitude = R / 2.0
    sin_weight_expanded = sin_weight[..., np.newaxis]
    sin_sq_delta_2 = np.divide(raw_magnitude, sin_weight_expanded, where=(sin_weight_expanded > 1e-3))
    sin_sq_delta_2 = np.where(sin_weight_expanded <= 1e-3, 0.0, sin_sq_delta_2)
    sin_sq_delta_2 = np.clip(sin_sq_delta_2, 0, 1)
    delta_wrap = 2 * np.arcsin(np.sqrt(sin_sq_delta_2))
    theta = theta_mean

    return theta, delta_wrap


def step2_only(data, S_i_hat=None):
    """
    Run Steps 1 & 2 only: Stokes extraction + wrapped retardance inversion.

    Parameters
    ----------
    data : ndarray, shape (H, W, n_wavelengths, 4)
    S_i_hat : array-like, optional

    Returns
    -------
    theta : ndarray, shape (H, W)
    delta_wrap : ndarray, shape (H, W, n_wavelengths)
    """
    H, W = data.shape[:2]

    I_0   = data[..., 0]
    I_45  = data[..., 1]
    I_90  = data[..., 2]
    I_135 = data[..., 3]

    S0, S1, S2 = compute_stokes_components(I_0, I_45, I_90, I_135)
    S1_hat, S2_hat = compute_normalised_stokes(S0, S1, S2)
    S_m_hat = np.stack([S1_hat, S2_hat], axis=-1)

    S_flat = S_m_hat.reshape(-1, S_m_hat.shape[-2], S_m_hat.shape[-1])

    if S_i_hat is None:
        S_i_hat = np.array([1.0, 0.0, 0.0])
    S_i_hat = _normalise_input_stokes_vector(S_i_hat)

    theta, delta_wrap = invert_wrapped_retardance(S_flat, S_i_hat)

    theta      = theta.reshape(H, W)
    delta_wrap = delta_wrap.reshape(H, W, -1)

    return theta, delta_wrap


# ── Circle fit ────────────────────────────────────────────────────────────────

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


# ── Interactive circle center picker ──────────────────────────────────────────

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
    circle_patch   = Circle((0, 0), radius=1, fill=False,
                             edgecolor="cyan", linewidth=2, visible=False)
    ax.add_patch(circle_patch)

    status_text = ax.text(
        0.01, 0.99,
        "Left-click circumference points (≥3). Right-click to undo.",
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
                    f"n={len(points)}  center=({cx:.1f}, {cy:.1f})  "
                    f"radius={r:.1f}px  ← click Done to accept"
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

    def _on_done(_event):
        if state["fit"] is None:
            status_text.set_text("Need a valid circle fit before clicking Done.")
            fig.canvas.draw_idle()
            return
        state["accepted"] = True
        plt.close(fig)

    def _on_reset(_event):
        points.clear()
        _update()

    done_ax   = fig.add_axes([0.80, 0.03, 0.16, 0.07])
    reset_ax  = fig.add_axes([0.62, 0.03, 0.16, 0.07])
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

    print(f"\nCircle center -> subpixel: ({cx:.2f}, {cy:.2f})  |  pixel address: col={ix}, row={iy}  |  radius: {r_px:.1f} px")
    return cx, cy, ix, iy, r_px


# ── Forward model ─────────────────────────────────────────────────────────────

def compute_del_sigma_predicted(C, delta_rad, wavelength_m, thickness_m):
    """
    Invert forward model to get predicted stress from measured retardance.

    Forward:   δ = C · Δσ · t · (2π / λ)
    Inversion: Δσ_predicted = δ · λ / (2π · C · t)
    """
    return (delta_rad * wavelength_m) / (2.0 * np.pi * C * thickness_m)


# ── Stress from input C vector ────────────────────────────────────────────────

def print_stress_from_input_C(delta_wrap_val, wavelengths_nm, thickness_m, C_input):
    """
    Using the C values supplied in the JSON5 config, compute and print
    del_sigma_predicted at the disk center for each wavelength.

    Parameters
    ----------
    delta_wrap_val : ndarray, shape (n_wavelengths,)
        Measured wrapped retardance in radians at disk center.
    wavelengths_nm : list of float
        Wavelengths in nm e.g. [650, 550, 450].
    thickness_m : float
        Specimen thickness in metres.
    C_input : list of float
        Stress-optic coefficients from JSON5 config, one per wavelength (1/Pa).
    """
    print("\n── Stress from input C (JSON5) ───────────────────────────────────")
    print(f"  Forward model:  Δσ = δ · λ / (2π · C · t)")
    print(f"  Δσ_analytical   = {DEL_SIGMA_ANALYTICAL / 1e6:.4f} MPa  [hardcoded]")
    print(f"  {'λ (nm)':>8}  {'C_input (1/Pa)':>16}  {'δ_meas (deg)':>14}  "
          f"{'Δσ_predicted (MPa)':>20}  {'error (MPa)':>12}")
    print(f"  {'-'*8}  {'-'*16}  {'-'*14}  {'-'*20}  {'-'*12}")

    for delta_rad, wl_nm, C in zip(delta_wrap_val, wavelengths_nm, C_input):
        wl_m    = wl_nm * 1e-9
        ds_pred = compute_del_sigma_predicted(C, delta_rad, wl_m, thickness_m)
        error   = ds_pred - DEL_SIGMA_ANALYTICAL
        print(f"  {wl_nm:>8}  {C:>16.4e}  {np.degrees(delta_rad):>14.4f}  "
              f"{ds_pred / 1e6:>20.4f}  {error / 1e6:>+12.4f}")

    print("─────────────────────────────────────────────────────────────────\n")


# ── C sweep & RMSE plot ───────────────────────────────────────────────────────

def plot_rmse_vs_C(delta_wrap_val, wavelengths_nm, thickness_m):
    """
    For each wavelength, sweep C over C_BOUNDS and plot
    RMSE = |Δσ_predicted(C) - Δσ_analytical|.

    Parameters
    ----------
    delta_wrap_val : ndarray, shape (n_wavelengths,)
    wavelengths_nm : list of float
    thickness_m : float
    """
    C_grid = np.logspace(
        np.log10(C_BOUNDS[0]),
        np.log10(C_BOUNDS[1]),
        num=N_POINTS,
    )

    n_wl  = len(wavelengths_nm)
    fig, axes = plt.subplots(1, n_wl, figsize=(7 * n_wl, 5))
    if n_wl == 1:
        axes = [axes]

    fig.suptitle(
        f"RMSE vs C   [Δσ_analytical = {DEL_SIGMA_ANALYTICAL:.4e} Pa]",
        fontsize=13, y=1.02,
    )

    print("\n── C sweep (RMSE in stress space) ────────────────────────────────")
    print(f"  Δσ_analytical = {DEL_SIGMA_ANALYTICAL / 1e6:.4f} MPa  [hardcoded]")
    print(f"  C_bounds      = [{C_BOUNDS[0]:.2e}, {C_BOUNDS[1]:.2e}] 1/Pa")
    print(f"  N_points      = {N_POINTS}")

    for ax, delta_rad, wl_nm in zip(axes, delta_wrap_val, wavelengths_nm):
        wl_m = wl_nm * 1e-9

        del_sigma_pred = compute_del_sigma_predicted(C_grid, delta_rad, wl_m, thickness_m)
        rmse           = np.sqrt((DEL_SIGMA_ANALYTICAL - del_sigma_pred) ** 2)

        i_min = int(np.argmin(rmse))
        C_opt = C_grid[i_min]
        r_min = rmse[i_min]

        ax.loglog(C_grid, rmse, color="steelblue", linewidth=1.5)
        ax.axvline(C_opt, color="red",    linestyle="--", linewidth=1.5,
                   label=f"C_opt = {C_opt:.3e} 1/Pa")
        ax.axhline(max(r_min, 1e-10), color="orange", linestyle=":", linewidth=1.2,
                   label=f"min RMSE = {r_min:.4e} Pa")
        ax.scatter([C_opt], [max(r_min, 1e-10)], color="red", zorder=5, s=60)

        ax.set_xscale("log")
        ax.set_xlabel("C  (1/Pa)", fontsize=11)
        ax.set_ylabel("RMSE  (Pa)  [log]", fontsize=11)
        ax.set_title(f"λ = {wl_nm} nm", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

        ds_opt = compute_del_sigma_predicted(C_opt, delta_rad, wl_m, thickness_m)
        print(f"\n  λ = {wl_nm} nm")
        print(f"    δ_measured          = {np.degrees(delta_rad):.4f} deg  ({delta_rad:.6f} rad)")
        print(f"    C_opt (grid min)    = {C_opt:.6e} 1/Pa")
        print(f"    Δσ_predicted(C_opt) = {ds_opt / 1e6:.4f} MPa")
        print(f"    Δσ_analytical       = {DEL_SIGMA_ANALYTICAL / 1e6:.4f} MPa")
        print(f"    min RMSE            = {r_min / 1e6:.6f} MPa")

    print("─────────────────────────────────────────────────────────────────\n")

    plt.tight_layout()
    plt.show()


# ── Diameter-profile C optimisation ──────────────────────────────────────────

def optimise_C_from_diameter(delta_wrap, cx, cy, r_px, wavelengths_nm, thickness_m):
    """
    Find the optimal stress-optic coefficient C (per wavelength) by matching
    the predicted Δσ profile along the horizontal diameter to the analytical
    solution for a diametrically loaded disk.

    Forward model: Δσ_pred(x) = f(x) / C,  where f(x) = δ(x)·λ/(2π·t)

    Minimising Σ_x (Δσ_ana(x) − f(x)/C)² yields the closed-form:

        C_opt = Σ f(x)² / Σ (Δσ_ana(x) · f(x))

    Parameters
    ----------
    delta_wrap   : ndarray [H, W, n_wl]   wrapped retardance in radians
    cx, cy       : float                  subpixel disk centre (pixels)
    r_px         : float                  disk radius in pixels
    wavelengths_nm : list of float        wavelengths in nm
    thickness_m  : float                  specimen thickness in metres

    Returns
    -------
    C_opt : ndarray [n_wl]   optimal C per wavelength (1/Pa)
    """
    scale_m_per_px = R_DISK / r_px
    iy = int(np.clip(round(cy), 0, delta_wrap.shape[0] - 1))
    W_img = delta_wrap.shape[1]

    col_min = max(0,     int(np.floor(cx - r_px)))
    col_max = min(W_img, int(np.ceil(cx  + r_px)) + 1)
    cols   = np.arange(col_min, col_max)
    x_phys = (cols - cx) * scale_m_per_px

    mask   = np.abs(x_phys) < R_DISK
    cols   = cols[mask]
    x_phys = x_phys[mask]

    ds_ana = np.array([del_sigma_disk(P_DISK, R_DISK, H_DISK, xp, 0.0) for xp in x_phys])

    n_wl   = len(wavelengths_nm)
    C_opt  = np.zeros(n_wl)

    for i, wl_nm in enumerate(wavelengths_nm):
        wl_m = wl_nm * 1e-9
        # f(x) = δ(x)·λ/(2π·t)  →  Δσ_pred = f/C
        f = delta_wrap[iy, cols, i] * wl_m / (2.0 * np.pi * thickness_m)
        C_opt[i] = np.sum(f ** 2) / np.sum(ds_ana * f)

    return C_opt


def plot_diameter_C_optimisation(delta_wrap, cx, cy, r_px, wavelengths_nm, thickness_m):
    """
    Optimise C from the diameter profile and plot analytical vs predicted Δσ
    before and after optimisation.
    """
    scale_m_per_px = R_DISK / r_px
    iy = int(np.clip(round(cy), 0, delta_wrap.shape[0] - 1))
    W_img = delta_wrap.shape[1]

    col_min = max(0,     int(np.floor(cx - r_px)))
    col_max = min(W_img, int(np.ceil(cx  + r_px)) + 1)
    cols   = np.arange(col_min, col_max)
    x_phys = (cols - cx) * scale_m_per_px

    mask   = np.abs(x_phys) < R_DISK
    cols   = cols[mask]
    x_phys = x_phys[mask]
    x_norm = x_phys / R_DISK

    ds_ana = np.array([del_sigma_disk(P_DISK, R_DISK, H_DISK, xp, 0.0) for xp in x_phys])

    C_opt = optimise_C_from_diameter(delta_wrap, cx, cy, r_px, wavelengths_nm, thickness_m)

    c_str = np.array2string(C_opt, formatter={"float_kind": lambda x: f"{x:.6e}"}, separator=" , ")
    print("\n── Diameter-profile C optimisation ───────────────────────────────")
    print(f"  {'λ (nm)':>8}  {'C_opt (1/Pa)':>14}")
    print(f"  {'-'*8}  {'-'*14}")
    for wl_nm, C in zip(wavelengths_nm, C_opt):
        print(f"  {wl_nm:>8}  {C:>14.6e}")
    print(f"\n  C : {c_str}")
    print("─────────────────────────────────────────────────────────────────\n")

    n_wl = len(wavelengths_nm)
    fig, axes = plt.subplots(1, n_wl, figsize=(7 * n_wl, 5), squeeze=False)
    fig.suptitle("Diameter-profile C optimisation — analytical vs predicted Δσ", fontsize=13)

    for ax, wl_nm, C in zip(axes[0], wavelengths_nm, C_opt):
        i = wavelengths_nm.index(wl_nm)
        wl_m = wl_nm * 1e-9
        f = delta_wrap[iy, cols, i] * wl_m / (2.0 * np.pi * thickness_m)
        ds_pred = f / C

        ax.scatter(x_norm, ds_pred, s=20, color="steelblue", alpha=0.7,
                   label=f"Predicted  (C={C:.3e})")
        ax.scatter(x_norm, ds_ana,  s=20, color="tomato",    alpha=0.7,
                   label="Analytical")
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("x / R", fontsize=11)
        ax.set_ylabel("Δσ  (Pa)", fontsize=11)
        ax.set_title(f"λ = {wl_nm} nm   C_opt = {C:.3e} 1/Pa", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return C_opt


# ── C_opt at specific disk points ─────────────────────────────────────────────

def print_c_opt_at_points(delta_wrap, cx, cy, r_px, wavelengths_nm, thickness_m):
    """
    Compute and print C_opt at specific physical (x, y) positions on the disk.

    For each point: C_opt = δ·λ / (2π·Δσ_analytical·t)
    Uses a 3×3 patch average of delta_wrap at the corresponding pixel.
    """
    points = [
        (0.0,           0.0,          "center (0, 0)"),
        (R_DISK / 3,    R_DISK / 3,   "+R/3, +R/3"),
        (R_DISK / 4,    R_DISK / 4,   "+R/4, +R/4"),
        (-R_DISK / 4,  -R_DISK / 4,   "-R/4, -R/4"),
    ]

    scale = R_DISK / r_px          # m per pixel
    H, W  = delta_wrap.shape[:2]
    wavelengths_m = [wl * 1e-9 for wl in wavelengths_nm]
    n_wl = len(wavelengths_nm)

    print("\n── C_opt at specific disk points ─────────────────────────────────────")
    header = f"  {'Point':>18}  {'Δσ_ana (MPa)':>14}"
    for wl in wavelengths_nm:
        header += f"  {'C@'+str(wl)+'nm (1/Pa)':>20}"
    print(header)
    print(f"  {'-'*18}  {'-'*14}" + f"  {'-'*20}" * n_wl)

    for x_m, y_m, label in points:
        ds_ana = del_sigma_disk(P_DISK, R_DISK, H_DISK, x_m, y_m)

        # Physical coords → pixel (row increases in same direction as +y here)
        col = cx + x_m / scale
        row = cy + y_m / scale
        r0 = int(np.clip(round(row) - 1, 0, H - 1))
        r1 = int(np.clip(round(row) + 2, 1, H))
        c0 = int(np.clip(round(col) - 1, 0, W - 1))
        c1 = int(np.clip(round(col) + 2, 1, W))

        dw_avg = delta_wrap[r0:r1, c0:c1, :].mean(axis=(0, 1))

        row_str = f"  {label:>18}  {ds_ana / 1e6:>14.4f}"
        for wl_m, dw in zip(wavelengths_m, dw_avg):
            if ds_ana > 1.0:
                c_opt = (dw * wl_m) / (2.0 * np.pi * ds_ana * thickness_m)
            else:
                c_opt = float("nan")
            row_str += f"  {c_opt:>20.6e}"
        print(row_str)

    print("──────────────────────────────────────────────────────────────────────\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def image_to_step2(params, output_dir="."):
    params = _merge_params_with_calibration(params)

    if "folderName" in params:
        data, metadata = photoelastimetry.io.load_raw(params["folderName"])
    elif "input_filename" in params:
        data, metadata = photoelastimetry.io.load_image(params["input_filename"])
    else:
        raise ValueError("Either 'folderName' or 'input_filename' must be specified in params.")

    profile = params.get("_calibration_profile")
    if profile is not None:
        data = photoelastimetry.calibrate.apply_blank_correction(
            data, profile["blank_correction"]
        )

    if params.get("crop") is not None:
        x1, x2, y1, y2 = params["crop"]
        data = data[y1:y2, x1:x2, :, :]

    if params.get("binning") is not None:
        binning = params["binning"]
        data = photoelastimetry.io.bin_image(data, binning)
        metadata["height"] //= binning
        metadata["width"]  //= binning

    if params.get("debug", False):
        photoelastimetry.plotting.show_all_channels(data, metadata)

    # ── Circle center: use value from JSON if present, else interactive ───────
    if "circle_center" in params:
        cx, cy, r_px = params["circle_center"]
        ix = int(round(cx)); iy = int(round(cy))
    else:
        cx, cy, ix, iy, r_px = pick_circle_center(data)

    if "S_i_hat" not in params:
        raise ValueError("Missing S_i_hat. Provide it directly or via calibration_file.")
    S_I_HAT = np.array(params["S_i_hat"], dtype=float)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    print("Running step2_only (Stokes inversion)...")
    theta, delta_wrap = step2_only(data, S_i_hat=S_I_HAT)

    print("theta     — min:", np.nanmin(theta),      "max:", np.nanmax(theta))
    print("delta_wrap— min:", np.nanmin(delta_wrap), "max:", np.nanmax(delta_wrap))

    # ── Single-pixel extraction ───────────────────────────────────────────────
    theta_val_single      = theta[iy, ix]
    delta_wrap_val_single = delta_wrap[iy, ix, :]
    if not np.all(np.isfinite(delta_wrap_val_single)):
        print(f"  WARNING: NaN/Inf in delta_wrap at center pixel ({ix},{iy}): {delta_wrap_val_single}")

    # ── 3×3 patch average (centred on disk center) ────────────────────────────
    r0, r1 = iy - 1, iy + 2   # rows: iy-1, iy, iy+1
    c0, c1 = ix - 1, ix + 2   # cols: ix-1, ix, ix+1
    theta_val      = theta     [r0:r1, c0:c1].mean()
    delta_wrap_val = delta_wrap[r0:r1, c0:c1, :].mean(axis=(0, 1))
    print(f"  (3×3 patch average: rows {r0}–{r1-1}, cols {c0}–{c1-1}  →  9 px)")

    wavelengths = params.get("wavelengths", [f"ch{i}" for i in range(delta_wrap_val.shape[0])])
    thickness   = float(params["thickness"])

    # ── Print δ — single pixel ────────────────────────────────────────────────
    print(f"\n── δ at disk center — single pixel (col={ix}, row={iy}) ──────────")
    print(f"  theta = {np.degrees(theta_val_single):.4f} deg  ({theta_val_single:.6f} rad)")
    for dv, wl in zip(delta_wrap_val_single, wavelengths):
        print(f"  δ [{wl}nm] = {np.degrees(dv):.4f} deg  ({dv:.6f} rad)")
    print("─────────────────────────────────────────────────────────────────\n")

    # ── Print δ — 3×3 patch average ───────────────────────────────────────────
    print(f"── δ at disk center — 3×3 patch average ─────────────────────────")
    print(f"  theta = {np.degrees(theta_val):.4f} deg  ({theta_val:.6f} rad)")
    for dv, wl in zip(delta_wrap_val, wavelengths):
        print(f"  δ [{wl}nm] = {np.degrees(dv):.4f} deg  ({dv:.6f} rad)")
    print("─────────────────────────────────────────────────────────────────\n")

    # ── Comparison plot: single pixel vs 3×3 patch average ───────────────────
    n_wl = len(wavelengths)
    fig, axes = plt.subplots(1, n_wl, figsize=(6 * n_wl, 4))
    if n_wl == 1:
        axes = [axes]
    fig.suptitle(
        f"δ at disk center — single pixel vs 3×3 patch average\n"
        f"center: col={ix}, row={iy}    patch: rows {r0}–{r1-1}, cols {c0}–{c1-1}",
        fontsize=11,
    )
    for ax, wl, d_single, d_avg in zip(axes, wavelengths, delta_wrap_val_single, delta_wrap_val):
        vals   = [np.degrees(d_single), np.degrees(d_avg)]
        labels = ["Single pixel", "3×3 average"]
        colors = ["steelblue", "tomato"]
        bars   = ax.bar(labels, vals, color=colors, width=0.4, edgecolor="k", linewidth=0.8)
        for bar, v in zip(bars, vals):
            label_y = v if np.isfinite(v) else 0.0
            ax.text(bar.get_x() + bar.get_width() / 2, label_y + 0.3,
                    f"{v:.3f}°", ha="center", va="bottom", fontsize=10)
        ax.set_title(f"λ = {wl} nm", fontsize=11)
        ax.set_ylabel("δ (degrees)", fontsize=10)
        finite_max = np.nanmax([v for v in vals if np.isfinite(v)]) if any(np.isfinite(v) for v in vals) else None
        if finite_max is not None and finite_max > 0:
            ax.set_ylim(0, finite_max * 1.15)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ── Step A: stress from input C (JSON5) using patch-averaged δ ───────────
    C_input = params.get("C")
    if C_input is None:
        print("Warning: 'C' not found in JSON5 params — skipping input-C stress calculation.")
    else:
        C_input = list(C_input)
        if len(C_input) != len(wavelengths):
            print(f"Warning: len(C)={len(C_input)} != len(wavelengths)={len(wavelengths)} "
                  f"— skipping input-C stress calculation.")
        else:
            print_stress_from_input_C(delta_wrap_val, wavelengths, thickness, C_input)

    # ── Step B: C sweep & RMSE plot using patch-averaged δ ───────────────────
    plot_rmse_vs_C(delta_wrap_val, wavelengths, thickness)

    # ── C_opt at specific disk points ─────────────────────────────────────────
    print_c_opt_at_points(delta_wrap, cx, cy, r_px, wavelengths, thickness)

    # ── Δσ map from step-2 results (using input C, first fringe order) ─────────
    if C_input is not None and len(C_input) == len(wavelengths):
        wavelengths_m = [wl * 1e-9 for wl in wavelengths]
        delta_sigma_channels = np.stack([
            (delta_wrap[..., i] * wl_m) / (2.0 * np.pi * C * thickness)
            for i, (wl_m, C) in enumerate(zip(wavelengths_m, C_input))
        ], axis=-1)
        delta_sigma_map = np.median(delta_sigma_channels, axis=-1)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(delta_sigma_map, cmap="viridis")
        ax.scatter([ix], [iy], c="red", s=80, marker="+", linewidths=2, zorder=5,
                   label=f"center ({ix},{iy})")
        fig.colorbar(im, ax=ax, label="Δσ (Pa)")
        ax.set_title("Principal stress difference  Δσ  (wrapped δ, input C)")
        ax.legend(fontsize=9)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

        ds_path = os.path.join(output_dir, "stress.tiff")
        os.makedirs(output_dir, exist_ok=True)
        tifffile.imwrite(ds_path, delta_sigma_map.astype(np.float32))
        print(f"Saved: {ds_path}")

        # ── Diameter scatter: analytical vs experimental Δσ ──────────────────
        scale_m_per_px = R_DISK / r_px          # metres per pixel

        # Horizontal diameter: all columns across the disk at row iy
        W_img = delta_wrap.shape[1]
        col_min = max(0,     int(np.floor(cx - r_px)))
        col_max = min(W_img, int(np.ceil(cx  + r_px)) + 1)
        cols = np.arange(col_min, col_max)

        # Physical x coordinate (m) relative to disk centre; y = 0 (horizontal)
        x_phys = (cols - cx) * scale_m_per_px

        # Only keep points strictly inside the disk
        mask = np.abs(x_phys) < R_DISK
        cols   = cols[mask]
        x_phys = x_phys[mask]

        # Experimental Δσ: median over wavelengths at row iy
        ds_exp = np.median(
            np.stack([
                (delta_wrap[iy, cols, i] * wl_m) / (2.0 * np.pi * C * thickness)
                for i, (wl_m, C) in enumerate(zip(wavelengths_m, C_input))
            ], axis=0),
            axis=0,
        )

        # Analytical Δσ along horizontal diameter (y_phys = 0)
        ds_ana = np.array([del_sigma_disk(P_DISK, R_DISK, H_DISK, xp, 0.0) for xp in x_phys])

        # Normalised x axis (−1 … +1 across diameter)
        x_norm = x_phys / R_DISK

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x_norm, ds_exp, s=20, color="steelblue", alpha=0.7,
                   label="Experimental (wrapped δ, input C)")
        ax.scatter(x_norm, ds_ana, s=20, color="tomato", alpha=0.7,
                   label="Analytical (Jaeger & Cook)")
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Normalised position along diameter  (x / R)", fontsize=11)
        ax.set_ylabel("Δσ  (Pa)", fontsize=11)
        ax.set_title("Principal stress difference along horizontal diameter", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    theta_path      = os.path.join(output_dir, "theta.tiff")
    delta_wrap_path = os.path.join(output_dir, "delta_wrap.tiff")

    tifffile.imwrite(theta_path,      theta.astype(np.float32))
    tifffile.imwrite(delta_wrap_path, delta_wrap.astype(np.float32))

    print(f"Saved: {theta_path}")
    print(f"Saved: {delta_wrap_path}")

    return theta, delta_wrap


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run photoelastic pipeline up to Step 2 and save delta/theta TIFFs."
    )
    parser.add_argument("json_filename", type=str, help="Path to the JSON5 parameter file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save output TIFFs (default: current directory).",
    )
    args = parser.parse_args()

    params = json5.load(open(args.json_filename, "r"))
    theta, delta_wrap = image_to_step2(params, output_dir=args.output_dir)

    print("theta shape:      ", theta.shape)
    print("delta_wrap shape: ", delta_wrap.shape)