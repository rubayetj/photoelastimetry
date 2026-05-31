"""
Calibration workflows for photoelastimetry experiments.

This module fits per-wavelength stress-optic coefficients (C), incoming
polarisation state (S_i_hat), and optional detector blank correction from a
multi-load calibration sequence.

Supported calibration methods:
- ``brazilian_disk``: diametrically-loaded disk with analytical stress field.
- ``coupon_test``: uniaxial coupon with nominal stress in a gauge ROI.
"""

import json
import os
import warnings
from copy import deepcopy
from datetime import datetime, timezone

import json5
import numpy as np
from scipy.optimize import least_squares

import photoelastimetry.io
from photoelastimetry.generate.disk import huang2014
from photoelastimetry.image import (
    compute_normalised_stokes,
    compute_stokes_components,
    predict_stokes,
    simulate_four_step_polarimetry,
)


def _normalise_wavelengths(wavelengths):
    """Return wavelengths as a 1D float array in meters."""
    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim == 0:
        wl = wl.reshape(1)
    if np.max(wl) > 1e-6:
        wl = wl * 1e-9
    return wl


def _identity_blank_correction(n_wavelengths):
    """Return identity blank correction coefficients."""
    return {
        "offset": np.zeros((n_wavelengths, 4), dtype=float).tolist(),
        "scale": np.ones((n_wavelengths, 4), dtype=float).tolist(),
        "mode": "identity",
    }


def _safe_relative_path(path, base_dir):
    """Convert path to relative form when possible."""
    if path is None:
        return None
    try:
        return os.path.relpath(path, start=base_dir)
    except ValueError:
        return path


def _as_float_array(value, expected_length=None, name="value"):
    """Coerce input to 1D float numpy array and optionally validate length."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if expected_length is not None:
        if arr.size == 1 and expected_length > 1:
            arr = np.full(expected_length, float(arr.item()), dtype=float)
        elif arr.size != expected_length:
            raise ValueError(f"{name} must have length {expected_length}, got {arr.size}.")
    return arr.astype(float)


def load_calibration_profile(profile_file):
    """
    Load and validate a calibration profile.

    Parameters
    ----------
    profile_file : str
        Path to calibration JSON/JSON5 profile.

    Returns
    -------
    dict
        Validated calibration profile.
    """
    if not os.path.exists(profile_file):
        raise ValueError(f"Calibration file not found: {profile_file}")

    with open(profile_file, "r") as f:
        profile = json5.load(f)

    required = {
        "version",
        "method",
        "wavelengths",
        "C",
        "S_i_hat",
        "blank_correction",
        "fit_metrics",
        "provenance",
    }
    missing = sorted(required.difference(profile.keys()))
    if missing:
        raise ValueError(f"Calibration profile missing required keys: {missing}")

    supported_methods = {"brazilian_disk", "coupon_test"}
    if profile["method"] not in supported_methods:
        raise ValueError(
            f"Unsupported calibration method '{profile['method']}'. "
            f"Supported methods: {sorted(supported_methods)}."
        )
    wavelengths = _normalise_wavelengths(profile["wavelengths"])
    C = _as_float_array(profile["C"], expected_length=wavelengths.size, name="C")

    S_i_hat = np.asarray(profile["S_i_hat"], dtype=float)
    if S_i_hat.size == 2:
        S_i_hat = np.append(S_i_hat, 0.0)
    if S_i_hat.size != 3:
        raise ValueError(f"S_i_hat must have length 2 or 3, got {S_i_hat.size}.")

    blank = profile["blank_correction"]
    if "offset" not in blank or "scale" not in blank:
        raise ValueError("blank_correction must contain 'offset' and 'scale'.")

    offset = np.asarray(blank["offset"], dtype=float)
    scale = np.asarray(blank["scale"], dtype=float)
    if offset.shape != (wavelengths.size, 4) or scale.shape != (wavelengths.size, 4):
        raise ValueError(
            "blank_correction offset/scale must have shape "
            f"({wavelengths.size}, 4); got {offset.shape} and {scale.shape}."
        )

    validated = dict(profile)
    validated["wavelengths"] = wavelengths.tolist()
    validated["C"] = C.tolist()
    validated["S_i_hat"] = S_i_hat.tolist()
    return validated


def compute_blank_correction(dark_frame, blank_frame, eps=1e-6):
    """
    Compute per-channel/per-angle scalar blank correction coefficients.

    Parameters
    ----------
    dark_frame : ndarray
        Dark reference image stack [H, W, C, 4].
    blank_frame : ndarray
        Blank reference image stack [H, W, C, 4].
    eps : float
        Minimum denominator for stability.

    Returns
    -------
    dict
        Dictionary with `offset` and `scale` arrays, each shape [C, 4].
    """
    dark = np.asarray(dark_frame, dtype=float)
    blank = np.asarray(blank_frame, dtype=float)

    if dark.shape != blank.shape:
        raise ValueError(f"Dark and blank frame shapes must match. Got {dark.shape} and {blank.shape}.")
    if dark.ndim != 4 or dark.shape[-1] != 4:
        raise ValueError(f"Blank correction frames must have shape [H, W, C, 4], got {dark.shape}.")

    offset = np.median(dark, axis=(0, 1))
    denom = np.median(blank, axis=(0, 1)) - offset

    if np.any(denom <= eps):
        raise ValueError("Blank correction denominator is non-positive for at least one channel/polariser.")

    scale = 1.0 / denom
    return {
        "offset": offset.tolist(),
        "scale": scale.tolist(),
        "mode": "dark_blank_scalar",
        "eps": float(eps),
    }


def apply_blank_correction(data, blank_correction):
    """
    Apply per-channel/per-angle blank correction to an image stack.

    Parameters
    ----------
    data : ndarray
        Image data [H, W, C, 4].
    blank_correction : dict
        Blank correction dictionary containing `offset` and `scale`.

    Returns
    -------
    ndarray
        Corrected image stack with same shape as input.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 4 or arr.shape[-1] != 4:
        raise ValueError(f"Image data must have shape [H, W, C, 4], got {arr.shape}.")

    if blank_correction is None:
        return arr

    if "offset" not in blank_correction or "scale" not in blank_correction:
        raise ValueError("blank_correction must contain 'offset' and 'scale'.")

    offset = np.asarray(blank_correction["offset"], dtype=float)
    scale = np.asarray(blank_correction["scale"], dtype=float)

    if offset.ndim != 2 or scale.ndim != 2 or offset.shape[1] != 4 or scale.shape[1] != 4:
        raise ValueError(
            "blank_correction offset/scale must each have shape [n_wavelengths, 4]. "
            f"Got {offset.shape} and {scale.shape}."
        )

    if offset.shape[0] != arr.shape[2] or scale.shape[0] != arr.shape[2]:
        raise ValueError(
            "blank_correction wavelength count must match data channels. "
            f"Got correction {offset.shape[0]} and data {arr.shape[2]}."
        )

    corrected = (arr - offset[np.newaxis, np.newaxis, :, :]) * scale[np.newaxis, np.newaxis, :, :]
    return corrected


def validate_calibration_config(config):
    """
    Validate and normalise calibration configuration.

    Parameters
    ----------
    config : dict
        Calibration configuration loaded from JSON5.

    Returns
    -------
    dict
        Normalised configuration with defaults.
    """
    method = config.get("method", "brazilian_disk")
    supported_methods = {"brazilian_disk", "coupon_test"}
    if method not in supported_methods:
        raise ValueError(f"Unsupported method='{method}'. Supported methods are {sorted(supported_methods)}.")

    if "wavelengths" not in config:
        raise ValueError("Calibration config must include 'wavelengths'.")
    wavelengths = _normalise_wavelengths(config["wavelengths"])

    if "thickness" not in config:
        raise ValueError("Calibration config must include 'thickness'.")
    thickness = float(config["thickness"])
    if thickness <= 0:
        raise ValueError("thickness must be positive.")

    geometry = dict(config.get("geometry", {}))
    if method == "brazilian_disk":
        for key in ("radius_m", "center_px", "pixels_per_meter"):
            if key not in geometry:
                raise ValueError(f"geometry must include '{key}' for method='brazilian_disk'.")

        radius_m = float(geometry["radius_m"])
        pixels_per_meter = float(geometry["pixels_per_meter"])
        center_px = np.asarray(geometry["center_px"], dtype=float)

        if radius_m <= 0:
            raise ValueError("geometry.radius_m must be positive.")
        if pixels_per_meter <= 0:
            raise ValueError("geometry.pixels_per_meter must be positive.")
        if center_px.size != 2:
            raise ValueError("geometry.center_px must have two elements [cx, cy].")

        edge_margin_fraction = float(geometry.get("edge_margin_fraction", 0.9))
        contact_exclusion_fraction = float(geometry.get("contact_exclusion_fraction", 0.12))
        if not (0 < edge_margin_fraction <= 1):
            raise ValueError("geometry.edge_margin_fraction must be in (0, 1].")
        if not (0 <= contact_exclusion_fraction < 1):
            raise ValueError("geometry.contact_exclusion_fraction must be in [0, 1).")

        geometry_validated = {
            "radius_m": radius_m,
            "center_px": center_px,
            "pixels_per_meter": pixels_per_meter,
            "edge_margin_fraction": edge_margin_fraction,
            "contact_exclusion_fraction": contact_exclusion_fraction,
        }
    else:
        for key in ("gauge_roi_px", "coupon_width_m"):
            if key not in geometry:
                raise ValueError(f"geometry must include '{key}' for method='coupon_test'.")

        gauge_roi_px = np.asarray(geometry["gauge_roi_px"], dtype=int)
        if gauge_roi_px.size != 4:
            raise ValueError("geometry.gauge_roi_px must be [x0, x1, y0, y1].")
        x0, x1, y0, y1 = [int(v) for v in gauge_roi_px]
        if x0 >= x1 or y0 >= y1:
            raise ValueError("geometry.gauge_roi_px must satisfy x0<x1 and y0<y1.")

        coupon_width_m = float(geometry["coupon_width_m"])
        if coupon_width_m <= 0:
            raise ValueError("geometry.coupon_width_m must be positive.")

        load_axis = str(geometry.get("load_axis", "x")).lower()
        if load_axis not in {"x", "y"}:
            raise ValueError("geometry.load_axis must be 'x' or 'y'.")

        transverse_stress_ratio = float(geometry.get("transverse_stress_ratio", 0.0))
        if not (-1.0 <= transverse_stress_ratio <= 1.0):
            raise ValueError("geometry.transverse_stress_ratio must be in [-1, 1].")

        roi_margin_px = int(geometry.get("roi_margin_px", 0))
        if roi_margin_px < 0:
            raise ValueError("geometry.roi_margin_px must be >= 0.")

        geometry_validated = {
            "gauge_roi_px": np.array([x0, x1, y0, y1], dtype=int),
            "coupon_width_m": coupon_width_m,
            "load_axis": load_axis,
            "transverse_stress_ratio": transverse_stress_ratio,
            "roi_margin_px": roi_margin_px,
        }

    load_steps = list(config.get("load_steps", []))
    if len(load_steps) == 0:
        raise ValueError("Calibration requires at least one load_step.")

    normalised_steps = []
    for step in load_steps:
        if "image_file" not in step or "load" not in step:
            raise ValueError("Each load step must include 'image_file' and 'load'.")
        image_file = step["image_file"]
        if not os.path.exists(image_file):
            raise ValueError(f"Calibration load-step image not found: {image_file}")
        normalised_steps.append({"image_file": image_file, "load": float(step["load"])})

    load_zero_tolerance = float(config.get("load_zero_tolerance", 1e-9))
    n_no_load = sum(abs(step["load"]) <= load_zero_tolerance for step in normalised_steps)
    n_loaded = sum(abs(step["load"]) > load_zero_tolerance for step in normalised_steps)
    if len(normalised_steps) < 4:
        warnings.warn(
            "Calibration usually benefits from at least 4 load_steps including a no-load image; "
            "continuing with the available data may reduce robustness.",
            stacklevel=2,
        )
    if n_no_load < 1:
        warnings.warn(
            "Calibration usually benefits from at least one no-load step (load ≈ 0); "
            "continuing without it will use a default initial S_i_hat guess unless one is provided in fit.initial_S_i_hat.",
            stacklevel=2,
        )
    if n_loaded < 3:
        warnings.warn(
            "Calibration usually benefits from at least three non-zero load steps; "
            "continuing with fewer loaded states may reduce identifiability and fit stability.",
            stacklevel=2,
        )

    fit_cfg = dict(config.get("fit", {}))
    fit_cfg.setdefault("max_points", 6000)
    fit_cfg.setdefault("loss", "soft_l1")
    fit_cfg.setdefault("f_scale", 0.05)
    fit_cfg.setdefault("max_nfev", 300)
    fit_cfg.setdefault("seed", 0)
    fit_cfg.setdefault("s3_identifiability_threshold", 0.02)
    fit_cfg.setdefault("prior_weight", 0.0)

    output_profile = config.get("output_profile", "calibration_profile.json5")
    output_report = config.get("output_report", "calibration_report.md")
    output_diagnostics = config.get("output_diagnostics", "calibration_diagnostics.npz")

    validated = {
        "S_i_hat": config.get("S_i_hat"),
        "method": method,
        "wavelengths": wavelengths,
        "thickness": thickness,
        "nu": float(config.get("nu", 1.0)),
        "geometry": geometry_validated,
        "load_steps": normalised_steps,
        "load_zero_tolerance": load_zero_tolerance,
        "dark_frame_file": config.get("dark_frame_file"),
        "blank_frame_file": config.get("blank_frame_file"),
        "fit": fit_cfg,
        "output_profile": output_profile,
        "output_report": output_report,
        "output_diagnostics": output_diagnostics,
        "provenance": dict(config.get("provenance", {})),
    }

    return validated


def _build_disk_coordinates(height, width, geometry):
    """Build metric X/Y coordinate grids and ROI masks from geometry config."""
    cx, cy = geometry["center_px"]
    ppm = geometry["pixels_per_meter"]
    R = geometry["radius_m"]

    x = (np.arange(width, dtype=float) - cx) / ppm
    y = (np.arange(height, dtype=float) - cy) / ppm
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2)
    disk_mask = r <= R

    core_mask = r <= geometry["edge_margin_fraction"] * R
    if geometry["contact_exclusion_fraction"] > 0:
        contact_r = geometry["contact_exclusion_fraction"] * R
        top_dist = np.sqrt(X**2 + (Y - R) ** 2)
        bottom_dist = np.sqrt(X**2 + (Y + R) ** 2)
        contact_mask = (top_dist < contact_r) | (bottom_dist < contact_r)
    else:
        contact_mask = np.zeros_like(core_mask)

    roi_mask = core_mask & (~contact_mask)
    return X, Y, disk_mask, roi_mask


def _build_coupon_masks(height, width, geometry):
    """Build coupon gauge masks from pixel ROI configuration."""
    x0, x1, y0, y1 = [int(v) for v in geometry["gauge_roi_px"]]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            f"geometry.gauge_roi_px {geometry['gauge_roi_px'].tolist()} is outside image bounds {(height, width)}."
        )

    roi_margin_px = int(geometry.get("roi_margin_px", 0))
    if (x1 - x0) <= 2 * roi_margin_px or (y1 - y0) <= 2 * roi_margin_px:
        raise ValueError("geometry.roi_margin_px is too large for the specified gauge_roi_px.")

    coupon_mask = np.zeros((height, width), dtype=bool)
    coupon_mask[y0:y1, x0:x1] = True

    roi_mask = np.zeros((height, width), dtype=bool)
    roi_mask[(y0 + roi_margin_px) : (y1 - roi_margin_px), (x0 + roi_margin_px) : (x1 - roi_margin_px)] = True

    return coupon_mask, roi_mask


def _coupon_stress_at_points(load, thickness, geometry, n_points):
    """Compute nominal coupon stress components at sampled points."""
    sigma_nominal = float(load) / (thickness * geometry["coupon_width_m"])
    transverse = geometry.get("transverse_stress_ratio", 0.0) * sigma_nominal
    axis = geometry.get("load_axis", "x")

    if axis == "x":
        sigma_xx = np.full(n_points, sigma_nominal, dtype=float)
        sigma_yy = np.full(n_points, transverse, dtype=float)
    else:
        sigma_xx = np.full(n_points, transverse, dtype=float)
        sigma_yy = np.full(n_points, sigma_nominal, dtype=float)
    sigma_xy = np.zeros(n_points, dtype=float)
    return sigma_xx, sigma_yy, sigma_xy


def _load_and_validate_image(path, expected_shape=None):
    """Load an image stack and validate expected dimensions."""
    data, _ = photoelastimetry.io.load_image(path)
    # Support raw 2D Bayer+polarisation frames by demosaicing to [H, W, C, 4].
    if data.ndim == 2:
        data = photoelastimetry.io.split_channels(data)
    if data.ndim == 4:
        data = photoelastimetry.io.collapse_duplicate_green_channel(data)
    data = np.asarray(data, dtype=float)

    if data.ndim != 4 or data.shape[-1] != 4:
        raise ValueError(
            f"Calibration images must have shape [H, W, n_wavelengths, 4]. Got {data.shape} for {path}."
        )

    if expected_shape is not None and data.shape != expected_shape:
        raise ValueError(
            f"All calibration images must share shape {expected_shape}; " f"got {data.shape} for {path}."
        )

    return data


def _build_preview_image(image):
    """Build a 2D preview image from a calibration image stack."""
    if image.ndim == 2:
        return image.astype(float)
    if image.ndim == 4:
        # Use I0 channel and average over wavelengths for a stable preview.
        return np.mean(image[..., 0], axis=2).astype(float)
    raise ValueError(f"Unsupported preview input shape: {image.shape}")


def _fit_circle_from_points(points):
    """Least-squares circle fit from Nx2 point coordinates."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("At least 3 [x, y] points are required to fit a circle.")

    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    params, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        raise ValueError("Selected points are degenerate; cannot fit a unique circle.")

    a, b_lin, c = params
    cx = -0.5 * a
    cy = -0.5 * b_lin
    r_sq = cx**2 + cy**2 - c
    if r_sq <= 0:
        raise ValueError("Fitted circle radius is not positive.")
    radius = float(np.sqrt(r_sq))
    return float(cx), float(cy), radius


def _disk_roi_pixel_count(height, width, geometry):
    """Return number of ROI pixels for a disk geometry candidate."""
    _, _, _, roi_mask = _build_disk_coordinates(height, width, geometry)
    return int(np.count_nonzero(roi_mask))


def interactive_geometry_wizard(config):
    """
    Interactively pick geometry from the first load-step image.

    - brazilian_disk: click center, then one edge point on the disk.
    - coupon_test: click top-left then bottom-right gauge ROI corners.
    """
    method = config.get("method", "brazilian_disk")
    if method not in {"brazilian_disk", "coupon_test"}:
        raise ValueError(f"Unsupported method='{method}' for interactive wizard.")

    steps = list(config.get("load_steps", []))
    if len(steps) == 0:
        raise ValueError("Interactive wizard requires at least one load step.")
    first_image = steps[0].get("image_file")
    if first_image is None:
        raise ValueError("First load step must include 'image_file'.")
    if not os.path.exists(first_image):
        raise ValueError(f"Load-step image not found: {first_image}")

    # Use the same preprocessing path as calibration dataset construction so
    # interactive geometry coordinates match model coordinates exactly.
    data = _load_and_validate_image(first_image, expected_shape=None)
    preview = _build_preview_image(data)

    import matplotlib.pyplot as plt

    cfg = deepcopy(config)
    geometry = dict(cfg.get("geometry", {}))

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(bottom=0.16)
    ax.imshow(preview, cmap="gray")
    ax.set_title("Calibration Geometry Wizard")
    ax.set_axis_off()
    if method == "brazilian_disk":
        from matplotlib.patches import Circle
        from matplotlib.widgets import Button

        ppm = geometry.get("pixels_per_meter")
        if ppm is None:
            raise ValueError(
                "geometry.pixels_per_meter is required for interactive disk calibration to convert pixels to meters."
            )
        ppm = float(ppm)
        if ppm <= 0:
            raise ValueError("geometry.pixels_per_meter must be positive.")
        instruction = ax.text(
            0.01,
            0.99,
            "Left-click circumference points (>=3). Right-click to undo.\nClick Done when the overlay circle matches.",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 4},
        )
        _ = instruction

        points = []
        point_scatter = ax.scatter([], [], c="yellow", s=24)
        circle_patch = Circle((0, 0), radius=1.0, fill=False, edgecolor="cyan", linewidth=2, visible=False)
        ax.add_patch(circle_patch)
        roi_contour = None
        fit_text = ax.text(
            0.01,
            0.05,
            "",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 3},
        )

        state = {"accepted": False, "fit": None}

        def _update_overlay():
            nonlocal roi_contour
            if len(points) == 0:
                point_scatter.set_offsets(np.empty((0, 2)))
            else:
                point_scatter.set_offsets(np.asarray(points, dtype=float))
            if len(points) >= 3:
                try:
                    cx, cy, radius_px = _fit_circle_from_points(points)
                    circle_patch.center = (cx, cy)
                    circle_patch.radius = radius_px
                    circle_patch.set_visible(True)
                    geom_candidate = {
                        "radius_m": radius_px / ppm,
                        "center_px": np.array([cx, cy], dtype=float),
                        "pixels_per_meter": ppm,
                        "edge_margin_fraction": float(geometry.get("edge_margin_fraction", 0.9)),
                        "contact_exclusion_fraction": float(geometry.get("contact_exclusion_fraction", 0.12)),
                    }
                    roi_pixels = _disk_roi_pixel_count(preview.shape[0], preview.shape[1], geom_candidate)
                    fit_text.set_text(
                        f"n={len(points)}  center=({cx:.1f}, {cy:.1f})  radius={radius_px:.1f}px  roi_pixels={roi_pixels}"
                    )
                    state["fit"] = (cx, cy, radius_px, roi_pixels)

                    if roi_contour is not None:
                        for coll in roi_contour.collections:
                            coll.remove()
                    _, _, _, roi_mask = _build_disk_coordinates(
                        preview.shape[0], preview.shape[1], geom_candidate
                    )
                    roi_contour = ax.contour(
                        roi_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.0
                    )
                except ValueError as exc:
                    circle_patch.set_visible(False)
                    fit_text.set_text(str(exc))
                    state["fit"] = None
                    if roi_contour is not None:
                        for coll in roi_contour.collections:
                            coll.remove()
                        roi_contour = None
            else:
                circle_patch.set_visible(False)
                fit_text.set_text("Need at least 3 points.")
                state["fit"] = None
                if roi_contour is not None:
                    for coll in roi_contour.collections:
                        coll.remove()
                    roi_contour = None
            fig.canvas.draw_idle()

        def _on_click(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            if event.button == 1:
                points.append((float(event.xdata), float(event.ydata)))
            elif event.button == 3 and len(points) > 0:
                points.pop()
            else:
                return
            _update_overlay()

        def _on_done(_event):
            if state["fit"] is None:
                fit_text.set_text("Need a valid circle fit before Done.")
                fig.canvas.draw_idle()
                return
            if state["fit"][3] <= 0:
                fit_text.set_text("ROI is empty for this circle. Add/adjust circumference points.")
                fig.canvas.draw_idle()
                return
            state["accepted"] = True
            plt.close(fig)

        def _on_reset(_event):
            points.clear()
            _update_overlay()

        done_ax = fig.add_axes([0.80, 0.03, 0.16, 0.07])
        done_btn = Button(done_ax, "Done")
        done_btn.on_clicked(_on_done)

        reset_ax = fig.add_axes([0.62, 0.03, 0.16, 0.07])
        reset_btn = Button(reset_ax, "Reset")
        reset_btn.on_clicked(_on_reset)

        fig.canvas.mpl_connect("button_press_event", _on_click)
        _update_overlay()
        plt.show()

        if not state["accepted"] or state["fit"] is None:
            raise ValueError("Interactive selection canceled. Circle was not accepted.")

        cx, cy, radius_px, _roi_pixels = state["fit"]
        geometry["center_px"] = [cx, cy]
        geometry["radius_m"] = radius_px / ppm
    else:
        points = plt.ginput(2, timeout=0)
        plt.close(fig)
        if len(points) != 2:
            raise ValueError("Interactive selection canceled. Please provide exactly two clicks.")
        (x0, y0), (x1, y1) = points
        xa, xb = sorted([int(round(x0)), int(round(x1))])
        ya, yb = sorted([int(round(y0)), int(round(y1))])
        if xa == xb or ya == yb:
            raise ValueError("Selected ROI must have non-zero width and height.")
        geometry["gauge_roi_px"] = [xa, xb, ya, yb]

    cfg["geometry"] = geometry
    return cfg


def _prepare_sampling_points(roi_mask, max_points, seed):
    """Select deterministic random sampling points from ROI mask."""
    y_idx, x_idx = np.where(roi_mask)
    n_total = y_idx.size
    if n_total == 0:
        raise ValueError("ROI mask is empty. Check geometry or ROI settings.")

    n_select = min(int(max_points), n_total)
    if n_select <= 0:
        raise ValueError("fit.max_points must be a positive integer.")

    rng = np.random.default_rng(seed)
    if n_select < n_total:
        indices = rng.choice(n_total, size=n_select, replace=False)
        y_idx = y_idx[indices]
        x_idx = x_idx[indices]

    return y_idx, x_idx


def _initial_s_i_hat_from_noload(measured_noload):
    """Estimate initial incoming Stokes state from no-load measurements."""
    s1 = float(np.median(measured_noload[..., 0]))
    s2 = float(np.median(measured_noload[..., 1]))
    magnitude_sq = s1**2 + s2**2
    s3 = float(np.sqrt(max(0.0, 1.0 - magnitude_sq)))
    s_i = np.array([s1, s2, s3], dtype=float)

    norm = np.linalg.norm(s_i)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if norm > 1.0:
        s_i = s_i / norm
    return s_i


def _default_initial_s_i_hat():
    """Return a conservative default incoming Stokes state guess."""
    return np.array([1.0, 0.0, 0.0], dtype=float)


def _build_dataset(config):
    """Load calibration images and construct regression dataset."""
    steps = config["load_steps"]
    expected_shape = None
    load_images = []
    loads = []

    for step in steps:
        image = _load_and_validate_image(step["image_file"], expected_shape=expected_shape)
        if expected_shape is None:
            expected_shape = image.shape
        load_images.append(image)
        loads.append(step["load"])

    H, W, n_channels, n_pols = expected_shape
    if n_pols != 4:
        raise ValueError("Calibration expects exactly 4 polariser images per channel.")

    if n_channels != config["wavelengths"].size:
        raise ValueError(
            "Number of image wavelength channels must match config wavelengths. "
            f"Got image channels={n_channels}, wavelengths={config['wavelengths'].size}."
        )

    dark_file = config["dark_frame_file"]
    blank_file = config["blank_frame_file"]
    if (dark_file is None) != (blank_file is None):
        raise ValueError("Provide both dark_frame_file and blank_frame_file, or neither.")

    if dark_file is not None:
        dark = _load_and_validate_image(dark_file, expected_shape=expected_shape)
        blank = _load_and_validate_image(blank_file, expected_shape=expected_shape)
        blank_correction = compute_blank_correction(dark, blank)
    else:
        blank_correction = _identity_blank_correction(n_channels)

    corrected_images = [apply_blank_correction(image, blank_correction) for image in load_images]

    if config["method"] == "brazilian_disk":
        X, Y, model_mask, roi_mask = _build_disk_coordinates(H, W, config["geometry"])
        roi_mask = roi_mask & model_mask
    elif config["method"] == "coupon_test":
        X = Y = None
        model_mask, roi_mask = _build_coupon_masks(H, W, config["geometry"])
    else:
        raise ValueError(f"Unsupported method '{config['method']}'.")

    y_idx, x_idx = _prepare_sampling_points(roi_mask, config["fit"]["max_points"], config["fit"]["seed"])

    measured_steps = []
    sigma_xx_steps = []
    sigma_yy_steps = []
    sigma_xy_steps = []

    no_load_tolerance = config["load_zero_tolerance"]
    no_load_measurements = []
    diagnostic_step_index = int(np.argmax(np.abs(loads)))
    diagnostic_image = corrected_images[diagnostic_step_index]

    for load, image in zip(loads, corrected_images):
        I0 = image[..., 0]
        I45 = image[..., 1]
        I90 = image[..., 2]
        I135 = image[..., 3]
        S0, S1, S2 = compute_stokes_components(I0, I45, I90, I135)
        S1_hat, S2_hat = compute_normalised_stokes(S0, S1, S2)

        measured = np.stack([S1_hat[y_idx, x_idx, :], S2_hat[y_idx, x_idx, :]], axis=-1)
        measured_steps.append(measured)

        if config["method"] == "brazilian_disk":
            sigma_xx, sigma_yy, sigma_xy = pointload_stress(
                X, Y, P=load, R=config["geometry"]["radius_m"], h=config["thickness"]
            )
            sigma_xx_steps.append(sigma_xx[y_idx, x_idx])
            sigma_yy_steps.append(sigma_yy[y_idx, x_idx])
            sigma_xy_steps.append(sigma_xy[y_idx, x_idx])
        else:
            sigma_xx, sigma_yy, sigma_xy = _coupon_stress_at_points(
                load, config["thickness"], config["geometry"], y_idx.size
            )
            sigma_xx_steps.append(sigma_xx)
            sigma_yy_steps.append(sigma_yy)
            sigma_xy_steps.append(sigma_xy)

        if abs(load) <= no_load_tolerance:
            no_load_measurements.append(measured)

    if len(no_load_measurements) == 0:
        warnings.warn(
            "No no-load measurements were found after processing load steps; "
            "using a default initial S_i_hat from the config file.",
            stacklevel=2,
        )
        # initial_s_i_hat = _default_initial_s_i_hat()
        initial_s_i_hat = config["S_i_hat"]
    else:
        noload = np.concatenate(no_load_measurements, axis=0)
        initial_s_i_hat = _initial_s_i_hat_from_noload(noload)

    dataset = {
        "method": config["method"],
        "wavelengths": config["wavelengths"],
        "nu": config["nu"],
        "thickness": config["thickness"],
        "loads": np.asarray(loads, dtype=float),
        "measured_steps": measured_steps,
        "sigma_xx_steps": sigma_xx_steps,
        "sigma_yy_steps": sigma_yy_steps,
        "sigma_xy_steps": sigma_xy_steps,
        "sample_y": y_idx,
        "sample_x": x_idx,
        "roi_mask": roi_mask,
        "model_mask": model_mask,
        # Backward-compatible key retained for previous diagnostics consumers.
        "disk_mask": model_mask,
        "blank_correction": blank_correction,
        "initial_s_i_hat": initial_s_i_hat,
        "X": X,
        "Y": Y,
        "geometry": config["geometry"],
        "diagnostic_step_index": diagnostic_step_index,
        "diagnostic_load": float(loads[diagnostic_step_index]),
        "diagnostic_image": diagnostic_image,
    }
    return dataset


def _normalise_s_i_hat(s_i_hat):
    """Project S_i_hat to the unit ball for physical consistency."""
    s_i = np.asarray(s_i_hat, dtype=float)
    norm = np.linalg.norm(s_i)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if norm > 1.0:
        return s_i / norm
    return s_i


def _decode_params(params, n_channels, fixed_s3=None):
    """Decode parameter vector into C and S_i_hat."""
    C = np.asarray(params[:n_channels], dtype=float)
    if fixed_s3 is None:
        s_i_hat = np.asarray(params[n_channels : n_channels + 3], dtype=float)
    else:
        s_i_hat = np.array([params[n_channels], params[n_channels + 1], fixed_s3], dtype=float)
    s_i_hat = _normalise_s_i_hat(s_i_hat)
    return C, s_i_hat


def calibration_residuals(params, dataset, fixed_s3=None):
    """
    Compute calibration residuals for least-squares fitting.

    Parameters
    ----------
    params : ndarray
        Parameter vector containing C values and S_i_hat components.
    dataset : dict
        Dataset dictionary from `_build_dataset`.
    fixed_s3 : float, optional
        If provided, S3 is fixed and only S1/S2 are optimised.

    Returns
    -------
    ndarray
        1D residual vector.
    """
    wavelengths = dataset["wavelengths"]
    n_channels = wavelengths.size
    C, s_i_hat = _decode_params(params, n_channels, fixed_s3=fixed_s3)

    residual_chunks = []
    for measured, sigma_xx, sigma_yy, sigma_xy in zip(
        dataset["measured_steps"],
        dataset["sigma_xx_steps"],
        dataset["sigma_yy_steps"],
        dataset["sigma_xy_steps"],
    ):
        for c, wl in enumerate(wavelengths):
            pred = predict_stokes(
                sigma_xx,
                sigma_yy,
                sigma_xy,
                C[c],
                dataset["nu"],
                dataset["thickness"],
                wl,
                s_i_hat,
            )
            residual_chunks.append((pred - measured[:, c, :]).ravel())

    if len(residual_chunks) == 0:
        return np.zeros(1, dtype=float)

    return np.concatenate(residual_chunks, axis=0)


def fit_calibration_parameters(dataset, fit_config):
    """
    Fit C and S_i_hat from calibration dataset.

    Parameters
    ----------
    dataset : dict
        Dataset dictionary from `_build_dataset`.
    fit_config : dict
        Fitting options.

    Returns
    -------
    dict
        Fit results containing estimated C, S_i_hat, metrics, and optimizer state.
    """
    wavelengths = dataset["wavelengths"]
    n_channels = wavelengths.size

    init_c = _as_float_array(
        fit_config.get("initial_C", np.full(n_channels, 3e-9)), expected_length=n_channels, name="initial_C"
    )
    init_s = np.asarray(fit_config.get("initial_S_i_hat", dataset["initial_s_i_hat"]), dtype=float)
    if init_s.size == 2:
        init_s = np.append(init_s, 0.0)
    if init_s.size != 3:
        raise ValueError(f"fit.initial_S_i_hat must have length 2 or 3, got {init_s.size}.")

    init_s = _normalise_s_i_hat(init_s)

    c_relative_bounds = fit_config.get("c_relative_bounds")
    if c_relative_bounds is not None:
        if len(c_relative_bounds) != 2:
            raise ValueError("fit.c_relative_bounds must be [lower_factor, upper_factor].")
        lower_factor = float(c_relative_bounds[0])
        upper_factor = float(c_relative_bounds[1])
        if lower_factor <= 0 or upper_factor <= 0 or lower_factor >= upper_factor:
            raise ValueError("fit.c_relative_bounds factors must satisfy 0 < lower < upper.")
        c_lower = np.maximum(init_c * lower_factor, 1e-15)
        c_upper = np.maximum(init_c * upper_factor, c_lower + 1e-15)
    else:
        c_lower = np.full(n_channels, 1e-15)
        c_upper = np.full(n_channels, 1e-4)

    x0_full = np.concatenate([init_c, init_s], axis=0)
    lb_full = np.concatenate([c_lower, np.full(3, -1.0)])
    ub_full = np.concatenate([c_upper, np.full(3, 1.0)])
    prior_weight = float(fit_config.get("prior_weight", 0.0))

    def residual_full(params):
        residual = calibration_residuals(params, dataset, fixed_s3=None)
        if prior_weight > 0:
            prior = prior_weight * (params - x0_full)
            return np.concatenate([residual, prior], axis=0)
        return residual

    result_full = least_squares(
        residual_full,
        x0_full,
        bounds=(lb_full, ub_full),
        loss=fit_config["loss"],
        f_scale=float(fit_config["f_scale"]),
        max_nfev=int(fit_config["max_nfev"]),
    )

    col_norms = (
        np.linalg.norm(result_full.jac, axis=0) if result_full.jac.size > 0 else np.zeros_like(x0_full)
    )
    reference_norm = np.median(col_norms[:-1]) if col_norms.size > 1 else 0.0
    s3_ratio = float(col_norms[-1] / max(reference_norm, 1e-12)) if col_norms.size > 0 else 0.0

    fallback = False
    fallback_reason = None
    threshold = float(fit_config.get("s3_identifiability_threshold", 0.02))

    if (not result_full.success) or (s3_ratio < threshold):
        fallback = True
        fallback_reason = "s3_identifiability" if s3_ratio < threshold else "full_fit_failed"

        if fallback_reason == "s3_identifiability":
            # If S3 is unidentifiable, restart fixed-S3 fit from the configured
            # initial guess to avoid inheriting unstable full-model updates.
            x0_fix = np.concatenate([init_c, init_s[:2]], axis=0)
        else:
            c_guess, s_guess = _decode_params(result_full.x, n_channels, fixed_s3=None)
            x0_fix = np.concatenate([c_guess, s_guess[:2]], axis=0)
        lb_fix = np.concatenate([c_lower, np.full(2, -1.0)])
        ub_fix = np.concatenate([c_upper, np.full(2, 1.0)])

        def residual_fixed(params):
            residual = calibration_residuals(params, dataset, fixed_s3=0.0)
            if prior_weight > 0:
                prior = prior_weight * (params - x0_fix)
                return np.concatenate([residual, prior], axis=0)
            return residual

        result = least_squares(
            residual_fixed,
            x0_fix,
            bounds=(lb_fix, ub_fix),
            loss=fit_config["loss"],
            f_scale=float(fit_config["f_scale"]),
            max_nfev=int(fit_config["max_nfev"]),
        )
        C_fit, S_fit = _decode_params(result.x, n_channels, fixed_s3=0.0)
    else:
        result = result_full
        C_fit, S_fit = _decode_params(result.x, n_channels, fixed_s3=None)

    residual = result.fun
    fit_metrics = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "cost": float(result.cost),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "n_residuals": int(residual.size),
        "n_samples": int(dataset["sample_y"].size),
        "n_load_steps": int(dataset["loads"].size),
        "fallback_used": bool(fallback),
        "fallback_reason": fallback_reason,
        "s3_identifiability_ratio": float(s3_ratio),
    }

    return {
        "C": C_fit,
        "S_i_hat": S_fit,
        "fit_metrics": fit_metrics,
        "optimizer_result": result,
    }


def _write_report(report_path, profile, config):
    """Write a concise markdown report for calibration outcomes."""
    lines = [
        "# Calibration Report",
        "",
        f"- Method: `{profile['method']}`",
        f"- Generated: `{profile['provenance']['generated_utc']}`",
        f"- Wavelengths (m): `{profile['wavelengths']}`",
        f"- C (1/Pa): `{profile['C']}`",
        f"- S_i_hat: `{profile['S_i_hat']}`",
        "",
        "## Fit Metrics",
        "",
    ]

    for key, value in profile["fit_metrics"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Geometry", ""])
    if config["method"] == "brazilian_disk":
        lines.extend(
            [
                f"- radius_m: `{config['geometry']['radius_m']}`",
                f"- center_px: `{config['geometry']['center_px'].tolist()}`",
                f"- pixels_per_meter: `{config['geometry']['pixels_per_meter']}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- gauge_roi_px: `{config['geometry']['gauge_roi_px'].tolist()}`",
                f"- coupon_width_m: `{config['geometry']['coupon_width_m']}`",
                f"- load_axis: `{config['geometry']['load_axis']}`",
                f"- transverse_stress_ratio: `{config['geometry']['transverse_stress_ratio']}`",
            ]
        )

    diagnostics_plot = profile.get("provenance", {}).get("diagnostics_plot_file")
    if diagnostics_plot is not None:
        lines.extend(["", "## Visual Diagnostics", "", f"- diagnostics_plot_file: `{diagnostics_plot}`"])

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _build_visual_diagnostics(dataset, fit_result):
    """Build measured/predicted diagnostic maps for one representative load step."""
    image = np.asarray(dataset["diagnostic_image"], dtype=float)
    load = float(dataset["diagnostic_load"])
    C = np.asarray(fit_result["C"], dtype=float)
    S_i_hat = np.asarray(fit_result["S_i_hat"], dtype=float)

    H, W, n_channels, _ = image.shape
    channel = int(min(1, n_channels - 1))

    if dataset["method"] == "brazilian_disk":
        X = dataset["X"]
        Y = dataset["Y"]
        sigma_xx, sigma_yy, sigma_xy = pointload_stress(
            X, Y, P=load, R=dataset["geometry"]["radius_m"], h=dataset["thickness"]
        )
    else:
        sigma_xx = np.zeros((H, W), dtype=float)
        sigma_yy = np.zeros((H, W), dtype=float)
        sigma_xy = np.zeros((H, W), dtype=float)
        x0, x1, y0, y1 = [int(v) for v in dataset["geometry"]["gauge_roi_px"]]
        sigma_nominal = load / (dataset["thickness"] * dataset["geometry"]["coupon_width_m"])
        transverse = dataset["geometry"].get("transverse_stress_ratio", 0.0) * sigma_nominal
        if dataset["geometry"].get("load_axis", "x") == "x":
            sigma_xx[y0:y1, x0:x1] = sigma_nominal
            sigma_yy[y0:y1, x0:x1] = transverse
        else:
            sigma_xx[y0:y1, x0:x1] = transverse
            sigma_yy[y0:y1, x0:x1] = sigma_nominal

    measured_i0 = image[:, :, channel, 0]
    predicted_s = predict_stokes(
        sigma_xx,
        sigma_yy,
        sigma_xy,
        C[channel],
        dataset["nu"],
        dataset["thickness"],
        dataset["wavelengths"][channel],
        S_i_hat,
    )
    pred_i0, _, _, _ = simulate_four_step_polarimetry(
        sigma_xx,
        sigma_yy,
        sigma_xy,
        C[channel],
        dataset["nu"],
        dataset["thickness"],
        dataset["wavelengths"][channel],
        S_i_hat,
        I0=1.0,
    )

    I0 = image[:, :, channel, 0]
    I45 = image[:, :, channel, 1]
    I90 = image[:, :, channel, 2]
    I135 = image[:, :, channel, 3]
    S0, S1, S2 = compute_stokes_components(I0, I45, I90, I135)
    measured_s1, measured_s2 = compute_normalised_stokes(S0, S1, S2)
    measured_s = np.stack([measured_s1, measured_s2], axis=-1)

    stokes_residual_mag = np.linalg.norm(predicted_s - measured_s, axis=-1)
    i0_residual_abs = np.abs(measured_i0 - pred_i0)
    roi_mask = np.asarray(dataset["roi_mask"], dtype=bool)

    def _mask_to_roi(arr):
        out = np.asarray(arr, dtype=float).copy()
        out[~roi_mask] = np.nan
        return out

    return {
        "channel": channel,
        "measured_i0": _mask_to_roi(measured_i0),
        "predicted_i0": _mask_to_roi(pred_i0),
        "i0_residual_abs": _mask_to_roi(i0_residual_abs),
        "measured_s1": _mask_to_roi(measured_s[..., 0]),
        "predicted_s1": _mask_to_roi(predicted_s[..., 0]),
        "stokes_residual_mag": _mask_to_roi(stokes_residual_mag),
    }


def _write_visual_diagnostics_plot(plot_path, dataset, fit_result):
    """Write a PNG summary of measured vs synthetic calibration fit quality."""
    maps = _build_visual_diagnostics(dataset, fit_result)
    roi_mask = np.asarray(dataset["roi_mask"], dtype=bool)

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=(13, 8), constrained_layout=True)
    _canvas = FigureCanvas(fig)
    axes = fig.subplots(2, 3)
    entries = [
        ("Measured I0", maps["measured_i0"], "gray"),
        ("Synthetic I0", maps["predicted_i0"], "gray"),
        ("|I0 residual|", maps["i0_residual_abs"], "magma"),
        ("Measured S1_hat", maps["measured_s1"], "coolwarm"),
        ("Synthetic S1_hat", maps["predicted_s1"], "coolwarm"),
        ("Stokes residual magnitude", maps["stokes_residual_mag"], "magma"),
    ]

    for ax, (title, arr, cmap) in zip(axes.ravel(), entries):
        im = ax.imshow(arr, cmap=cmap)
        ax.contour(roi_mask.astype(float), levels=[0.5], colors="w", linewidths=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"Calibration Fit Diagnostics (channel index={maps['channel']}, load={dataset['diagnostic_load']})",
        fontsize=12,
    )
    fig.savefig(plot_path, dpi=160)


def run_calibration(config):
    """
    Run full calibration workflow from config.

    Parameters
    ----------
    config : dict
        Calibration input configuration.

    Returns
    -------
    dict
        Result dictionary containing the calibration profile and metadata.
    """
    cfg = validate_calibration_config(config)
    dataset = _build_dataset(cfg)
    fit_result = fit_calibration_parameters(dataset, cfg["fit"])

    output_profile = cfg["output_profile"]
    output_report = cfg["output_report"]
    output_diagnostics = cfg["output_diagnostics"]
    report_root, report_ext = os.path.splitext(output_report)
    output_diagnostics_plot = f"{report_root}_fit.png" if report_ext else f"{output_report}_fit.png"

    output_dir = os.path.dirname(output_profile)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for path in (output_report, output_diagnostics, output_diagnostics_plot):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    profile_dir = os.path.dirname(output_profile) if os.path.dirname(output_profile) else os.getcwd()

    provenance_steps = [
        {
            "load": float(step["load"]),
            "image_file": _safe_relative_path(step["image_file"], profile_dir),
        }
        for step in cfg["load_steps"]
    ]

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dark_frame_file": _safe_relative_path(cfg["dark_frame_file"], profile_dir),
        "blank_frame_file": _safe_relative_path(cfg["blank_frame_file"], profile_dir),
        "load_steps": provenance_steps,
        "diagnostics_file": _safe_relative_path(output_diagnostics, profile_dir),
        "diagnostics_plot_file": _safe_relative_path(output_diagnostics_plot, profile_dir),
    }
    provenance.update(cfg["provenance"])

    profile = {
        "version": 1,
        "method": cfg["method"],
        "wavelengths": cfg["wavelengths"].tolist(),
        "C": fit_result["C"].tolist(),
        "S_i_hat": fit_result["S_i_hat"].tolist(),
        "blank_correction": dataset["blank_correction"],
        "fit_metrics": fit_result["fit_metrics"],
        "provenance": provenance,
    }

    with open(output_profile, "w") as f:
        json.dump(profile, f, indent=2)

    _write_visual_diagnostics_plot(output_diagnostics_plot, dataset, fit_result)
    _write_report(output_report, profile, cfg)
    visual_maps = _build_visual_diagnostics(dataset, fit_result)

    np.savez(
        output_diagnostics,
        roi_mask=dataset["roi_mask"],
        model_mask=dataset["model_mask"],
        disk_mask=dataset["disk_mask"],
        sample_y=dataset["sample_y"],
        sample_x=dataset["sample_x"],
        loads=dataset["loads"],
        C=np.asarray(profile["C"], dtype=float),
        S_i_hat=np.asarray(profile["S_i_hat"], dtype=float),
        residual=fit_result["optimizer_result"].fun,
        diagnostic_channel=np.array([visual_maps["channel"]], dtype=int),
        measured_i0=visual_maps["measured_i0"],
        predicted_i0=visual_maps["predicted_i0"],
        i0_residual_abs=visual_maps["i0_residual_abs"],
        measured_s1=visual_maps["measured_s1"],
        predicted_s1=visual_maps["predicted_s1"],
        stokes_residual_mag=visual_maps["stokes_residual_mag"],
    )

    return {
        "profile": profile,
        "profile_file": output_profile,
        "report_file": output_report,
        "diagnostics_file": output_diagnostics,
        "diagnostics_plot_file": output_diagnostics_plot,
    }
