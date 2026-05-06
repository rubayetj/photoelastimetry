import argparse
import os
import re
import json
import json5
import tifffile
import numpy as np
from scipy.ndimage import gaussian_filter  # kept in case you want smoothing
import scipy.io as sio  # Add this import at the top with other imports
import photoelastimetry.calibrate
import photoelastimetry.io
import photoelastimetry.optimise
import photoelastimetry.plotting
import photoelastimetry.seeding
from photoelastimetry.image import compute_retardance, simulate_four_step_polarimetry
from photoelastimetry.main import _merge_params_with_calibration, _normalise_wavelengths

def image_to_stress(params, output_filename=None):
    """
    Convert photoelastic images to stress maps.

    This function processes raw photoelastic data to recover stress distribution maps
    using the stress-optic law and polarisation analysis.

    Args:
        params (dict): Configuration dictionary containing:
            - input_filename (str, optional): Path to input image file. If None, raw images are loaded from folderName.
            - folderName (str): Path to folder containing raw photoelastic images
            - crop (list, optional): Crop region as [x1, x2, y1, y2]
            - debug (bool): If True, display all channels for debugging
            - C (float or list): Stress-optic coefficient(s) in 1/Pa
            - thickness (float): Sample thickness in meters
            - wavelengths (list): List of wavelengths in nanometers
            - S_i_hat (list): Incoming normalised Stokes vector [S1_hat, S2_hat, S3_hat]
            - seeding (dict, optional): Seeding controls (`enabled`, `n_max`, `sigma_max`)
            - top-level optimise options (optional): `knot_spacing`, `spline_degree`,
              `boundary_mask_file`, `boundary_values_files`, `boundary_weight`,
              `regularisation_weight` (or `regularization_weight`), `regularisation_order`,
              `external_potential_file`, `external_potential_gradient`, `max_iterations`,
              `tolerance`, `verbose`, `debug`
        output_filename (str, optional): Path to save the output stress map image.
            If None, the stress map is not saved. Can also be specified in params.

    Returns:
        numpy.ndarray: Stress map (principal stress difference) with same shape as delta.
    """

    # Merge with calibration profile if provided
    params = _merge_params_with_calibration(params)

    # Load data
    if "folderName" in params:
        data, metadata = photoelastimetry.io.load_raw(params["folderName"])
    elif "input_filename" in params:
        data, metadata = photoelastimetry.io.load_image(params["input_filename"])
    else:
        raise ValueError("Either 'folderName' or 'input_filename' must be specified in params.")

    # Apply blank correction from calibration profile if present
    profile = params.get("_calibration_profile")
    if profile is not None:
        data = photoelastimetry.calibrate.apply_blank_correction(
            data, profile["blank_correction"]
        )

    # Optional crop
    if params.get("crop") is not None:
        x1, x2, y1, y2 = params["crop"]
        data = data[y1:y2, x1:x2, :, :]
        if params.get("debug", False):
            photoelastimetry.io.save_image("debug_cropped_image.tiff", data, metadata)

    # Debug: save a single channel before binning
    if params.get("debug", False):
        tifffile.imwrite("debug_before_binning.tiff", data[:, :, 0, 0])

    # Optional binning
    if params.get("binning") is not None:
        binning = params["binning"]
        data = photoelastimetry.io.bin_image(data, binning)
        metadata["height"] //= binning
        metadata["width"] //= binning

    # Debug: show all channels
    if params.get("debug", False):
        photoelastimetry.plotting.show_all_channels(data, metadata)

    # Required physical parameters
    if "C" not in params:
        raise ValueError("Missing stress-optic coefficient 'C'. Provide it directly or via calibration_file.")
    if "thickness" not in params:
        raise ValueError("Missing sample thickness 'thickness'.")
    if "wavelengths" not in params:
        raise ValueError("Missing wavelengths. Provide them directly or via calibration_file.")
    if "S_i_hat" not in params:
        raise ValueError("Missing S_i_hat. Provide it directly or via calibration_file.")

    C = params["C"]              # 1/Pa
    L = params["thickness"]      # m
    WAVELENGTHS = _normalise_wavelengths(params["wavelengths"])
    NU = 1.0                     # solid sample

    # Normalize C to an array
    if isinstance(C, (list, np.ndarray)):
        C_VALUES = np.asarray(C, dtype=float)
    else:
        # if you know you always have 3 channels, keep 3 copies
        C_VALUES = np.array([C, C, C], dtype=float)

    # Incoming Stokes state
    S_I_HAT = np.array(params["S_i_hat"], dtype=float)
    if S_I_HAT.size == 2:
        S_I_HAT = np.append(S_I_HAT, 0.0)

    # Legacy solver configs are not supported
    if "solver" in params:
        raise ValueError(
            "`solver` is no longer supported. "
            "image_to_stress now always runs the optimise solver. "
            "Remove `solver` from params."
        )
    if "global_mean_stress" in params or "global_solver" in params:
        raise ValueError(
            "Nested solver config blocks (`global_mean_stress`, `global_solver`) are no longer supported. "
            "Move solver options to top-level params."
        )

    # Seeding config
    seeding_config = params.get("seeding", {})
    n_max = seeding_config.get("n_max", 6)
    sigma_max = seeding_config.get("sigma_max", 10e6)

    # Correction parameters
    correction_params = params.get("correction", {})

    # Phase decomposed seeding
    print("Running phase decomposed seeding...")
    seed = photoelastimetry.seeding.phase_decomposed_seeding(
        data,
        WAVELENGTHS,
        C_VALUES,
        NU,
        L,
        S_i_hat=S_I_HAT,
        sigma_max=sigma_max,
        n_max=n_max,
        correction_params=correction_params,
    )
    delta = seed.retardance
    theta = seed.theta

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(delta[:, :, 0], cmap="inferno")
    plt.colorbar(label="Retardation (radians)")

    plt.subplot(1, 2, 2)
    plt.imshow(theta, cmap="hsv")
    plt.colorbar(label="Principal Stress Angle (radians)")
    plt.show()

    # Save retardation field for inspection
    photoelastimetry.io.save_image("delta_final.tiff", delta, metadata)
    # Convert retardation to principal stress difference using stress–optic law
    # sigma_diff = delta * (2.0 * np.pi * C_VALUES * L / WAVELENGTHS)
    sigma_diff = seed.delta_sigma  # directly from seeding output
    print("C values = :", C_VALUES)
    # Save retardation field for inspection
    photoelastimetry.io.save_image("sigma_diff.tiff", sigma_diff, metadata)
    # Basic stats for sanity check
    print(
        "σ_diff stats:",
        "min =", np.nanmin(sigma_diff),
        "max =", np.nanmax(sigma_diff),
        "mean =", np.nanmean(sigma_diff),
    )

    # If requested, save stress map
    if output_filename is None and "output" in params:
        output_filename = params["output"]
    if output_filename is not None:
        photoelastimetry.io.save_image(output_filename, sigma_diff, metadata)
        print(f"Saved stress map to: {output_filename}")

    return sigma_diff


if __name__ == "__main__":
    """Command line interface for image_to_stress function."""
    parser = argparse.ArgumentParser(description="Convert photoelastic images to stress maps.")
    parser.add_argument(
        "json_filename",
        type=str,
        help="Path to the JSON5 parameter file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output stress map image (optional).",
    )
    args = parser.parse_args()

    # Load params and run
    params = json5.load(open(args.json_filename, "r"))
    sigma_diff = image_to_stress(params, output_filename=args.output)

    # Optional: inspect shape here
    print("Computed stress map shape:", sigma_diff.shape)
# End of file