import json5
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from tqdm import tqdm

from photoelastimetry.image import (
    compute_normalised_stokes,
    compute_stokes_components,
    simulate_four_step_polarimetry,
)
from photoelastimetry.plotting import virino

virino_cmap = virino()


# Standard Brazil test analytical solution
def diametrical_stress_cartesian(X, Y, P, R):
    """
    Exact Brazil test solution from ISRM standards and Jaeger & Cook
    P: total load (force per unit thickness)
    R: disk radius

    Key validation: At center (0,0):
    - sigma_x = 2P/(pi*R) (tensile)
    - sigma_y = -6P/(pi*R) (compressive)
    - tau_xy = 0
    """

    X_safe = X.copy()
    Y_safe = Y.copy()

    # Small offset to avoid singularities at origin
    origin_mask = (X**2 + Y**2) < (0.001 * R) ** 2
    X_safe = np.where(origin_mask, 0.001 * R, X_safe)
    Y_safe = np.where(origin_mask, 0.001 * R, Y_safe)

    # Distance from load points
    r1 = np.sqrt(X_safe**2 + (Y_safe - R) ** 2)  # from (0, R)
    r2 = np.sqrt(X_safe**2 + (Y_safe + R) ** 2)  # from (0, -R)

    # Angles from load points
    theta1 = np.arctan2(X_safe, Y_safe - R)
    theta2 = np.arctan2(X_safe, Y_safe + R)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_xx = (
            -(2 * P / np.pi)
            * (np.cos(theta1) ** 2 * (Y_safe - R) / (r1**2) - np.cos(theta2) ** 2 * (Y_safe + R) / (r2**2))
            / R
        )

        sigma_yy = (
            -(2 * P / np.pi)
            * (np.sin(theta1) ** 2 * (Y_safe - R) / (r1**2) - np.sin(theta2) ** 2 * (Y_safe + R) / (r2**2))
            / R
        )

        tau_xy = (
            -(2 * P / np.pi)
            * (
                np.sin(theta1) * np.cos(theta1) * (Y_safe - R) / (r1**2)
                - np.sin(theta2) * np.cos(theta2) * (Y_safe + R) / (r2**2)
            )
            / R
        )

    return sigma_xx, sigma_yy, tau_xy


def generate_synthetic_brazil_test(
    X,
    Y,
    P,
    R,
    S_i_hat,
    mask,
    wavelengths=None,
    thickness=None,
    C=None,
    polarisation_efficiency=1.0,
    **kwargs,
):
    """
    Generate synthetic Brazil test data for validation
    This function creates a synthetic dataset based on the analytical solution
    and saves it in a format suitable for testing.
    """

    # Backward compatibility: older callers used keyword `wavelengths_nm`
    if wavelengths is None:
        wavelengths = kwargs.pop("wavelengths_nm", None)
    elif "wavelengths_nm" in kwargs:
        raise TypeError("Pass only one of `wavelengths` or `wavelengths_nm`, not both.")

    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    if wavelengths is None:
        raise TypeError("Missing required wavelength input: pass `wavelengths` (meters).")

    # Get stress components directly
    sigma_xx, sigma_yy, tau_xy = diametrical_stress_cartesian(X, Y, P, R)

    # Mask outside the disk
    sigma_xx[~mask] = np.nan
    sigma_yy[~mask] = np.nan
    tau_xy[~mask] = np.nan

    # Principal stress difference and angle
    sigma_avg = 0.5 * (sigma_xx + sigma_yy)
    R_mohr = np.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + tau_xy**2)
    sigma1 = sigma_avg + R_mohr
    sigma2 = sigma_avg - R_mohr
    principal_diff = sigma1 - sigma2
    theta_p = 0.5 * np.arctan2(2 * tau_xy, sigma_xx - sigma_yy)

    # Mask again
    principal_diff[~mask] = np.nan
    theta_p[~mask] = np.nan

    height, width = sigma_xx.shape

    synthetic_images = np.empty((height, width, 3, 4))  # RGB, 4 polarizer angles

    # Use incoming light fully S1 polarized (standard setup)
    # S_i_hat = np.array([0.0, 0.0, 1.0])
    nu = 1.0  # Solid sample

    for i, lambda_light in tqdm(enumerate(wavelengths)):
        # Generate four-step polarimetry images using Mueller matrix approach
        I0_pol, I45_pol, I90_pol, I135_pol = simulate_four_step_polarimetry(
            sigma_xx, sigma_yy, tau_xy, C[i], nu, thickness, lambda_light, S_i_hat
        )

        synthetic_images[:, :, i, 0] = I0_pol
        synthetic_images[:, :, i, 1] = I45_pol
        synthetic_images[:, :, i, 2] = I90_pol
        synthetic_images[:, :, i, 3] = I135_pol

    return (
        synthetic_images,
        principal_diff,
        theta_p,
        sigma_xx,
        sigma_yy,
        tau_xy,
    )


def post_process_synthetic_data(  # pragma: no cover
    principal_diff, theta_p, sigma_xx, sigma_yy, tau_xy, S_i_hat, t_sample, C, lambda_light, outname
):
    plt.figure(figsize=(12, 12), layout="constrained")

    # Calculate retardation
    retardation = (2 * np.pi * t_sample * C * principal_diff) / lambda_light
    f_sigma = lambda_light / (2 * C * t_sample)  # material
    fringe_order = principal_diff / f_sigma  # N = (σ1 - σ2)/f_σ

    # Photoelastic parameters
    # For circular polariscope (dark field): I ∝ sin²(δ/2) where δ is retardation
    intensity_dark = np.sin(retardation / 2) ** 2  # Dark field intensity

    # For isoclinic lines, we need the extinction angle in plane polariscope
    isoclinic_angle = theta_p  # Principal stress angle (can be negative)

    # Generate four-step polarimetry images using Mueller matrix approach
    # Use incoming light fully S1 polarized (standard setup)

    nu = 1.0  # Solid sample
    I0_pol, I45_pol, I90_pol, I135_pol = simulate_four_step_polarimetry(
        sigma_xx, sigma_yy, tau_xy, C, nu, t_sample, lambda_light, S_i_hat
    )

    # Calculate Stokes parameters from polarimetry
    S0, S1, S2 = compute_stokes_components(I0_pol, I45_pol, I90_pol, I135_pol)
    S1_hat, S2_hat = compute_normalised_stokes(S0, S1, S2)

    # Degree of linear polarization
    DoLP = np.sqrt(S1_hat**2 + S2_hat**2)

    # Angle of linear polarization
    AoLP = np.mod(0.5 * np.arctan2(S2_hat, S1_hat), np.pi)

    # Plot characteristic Brazil test photoelastic patterns
    plt.clf()

    plt.subplot(4, 4, 1)
    # Plot fringe order with proper levels for Brazil test
    max_fringe = np.nanmax(fringe_order)
    levels = np.linspace(0, min(max_fringe, 8), 25)
    plt.contourf(X, Y, fringe_order, levels=levels, cmap="plasma", extend="max")
    plt.colorbar(label="Fringe Order N", shrink=0.8)
    plt.title("Isochromatic Fringes")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")
    # Add integer fringe contour lines (dark fringes)
    integer_levels = np.arange(0.5, min(max_fringe, 8), 1.0)
    plt.contour(
        X,
        Y,
        fringe_order,
        levels=integer_levels,
        colors="black",
        linewidths=1.0,
    )

    plt.subplot(4, 4, 2)
    # Dark field circular polariscope (what you actually see)
    plt.contourf(X, Y, intensity_dark, levels=50, cmap="gray")
    plt.colorbar(label="Intensity", shrink=0.8)
    plt.title("Dark Field Circular\nPolariscope")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 3)
    # Principal stress directions (isoclinics)
    isoclinic_angle_deg = np.rad2deg(isoclinic_angle)
    # Wrap to [-90, 90] for better visualization of stress directions
    isoclinic_angle_deg = ((isoclinic_angle_deg + 90) % 180) - 90
    plt.contourf(X, Y, isoclinic_angle_deg, levels=36, cmap=virino_cmap)
    plt.colorbar(label="Isoclinic Angle (°)", shrink=0.8)
    plt.title("Isoclinic Lines\n(Principal Stress Direction)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 4)
    plt.contourf(X, Y, DoLP, cmap="viridis")
    plt.colorbar(label="DoLP", shrink=0.8)
    plt.title("Degree of Linear\nPolarization")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 5)
    plt.contourf(X, Y, AoLP, levels=36, cmap=virino_cmap, vmin=0, vmax=np.pi)
    plt.colorbar(label="AoLP (rad)", shrink=0.8)
    plt.title("Angle of Linear\nPolarization")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    # Second row: Four-step polarimetry images (what you'd actually capture)
    polarizer_angles = ["0°", "45°", "90°", "135°"]
    polarimetry_images = [I0_pol, I45_pol, I90_pol, I135_pol]

    for i, (img, angle) in enumerate(zip(polarimetry_images, polarizer_angles)):
        plt.subplot(4, 4, 6 + i)
        plt.contourf(X, Y, img, levels=50, cmap="gray")
        plt.colorbar(label="Intensity", shrink=0.8)
        plt.title(f"Linear Polarizer at {angle}")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.gca().set_aspect("equal")

    # Add one more plot showing the difference between max and min intensities
    plt.subplot(4, 4, 10)
    intensity_range = np.maximum.reduce(polarimetry_images) - np.minimum.reduce(polarimetry_images)
    plt.contourf(X, Y, intensity_range, levels=50, cmap="hot")
    plt.colorbar(label="Intensity Range", shrink=0.8)
    plt.title("Polarimetric Contrast\n(Max - Min Intensity)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    # Third row: Stress components
    plt.subplot(4, 4, 11)
    sigma_xx_MPa = sigma_xx / 1e6  # Convert to MPa
    sigma_xx_max = np.nanmax(np.abs(sigma_xx_MPa))
    plt.pcolormesh(
        X,
        Y,
        sigma_xx_MPa,
        cmap="plasma",
        norm=LogNorm(vmin=sigma_xx_max / 1e3, vmax=sigma_xx_max),
        # norm=SymLogNorm(
        # linthresh=sigma_xx_max / 1e3, vmin=-sigma_xx_max, vmax=sigma_xx_max
        # ),
    )
    plt.colorbar(label="σ_xx (MPa)", shrink=0.8)
    plt.title("Horizontal Stress σ_xx")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 12)
    sigma_yy_MPa = sigma_yy / 1e6
    sigma_yy_max = np.nanmax(np.abs(sigma_yy_MPa))
    plt.pcolormesh(
        X,
        Y,
        sigma_yy_MPa,
        cmap="RdBu_r",
        norm=SymLogNorm(
            linthresh=sigma_yy_max / 1e3,
            vmin=-sigma_yy_max,
            vmax=sigma_yy_max,
        ),
    )
    plt.colorbar(label="σ_yy (MPa)", shrink=0.8)
    plt.title("Vertical Stress σ_yy")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 13)
    tau_xy_MPa = tau_xy / 1e6
    tau_xy_max = np.nanmax(np.abs(tau_xy_MPa))
    plt.pcolormesh(
        X,
        Y,
        tau_xy_MPa,
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=tau_xy_max / 1e6, vmin=-tau_xy_max, vmax=tau_xy_max),
    )
    plt.colorbar(label="τ_xy (MPa)", shrink=0.8)
    plt.title("Shear Stress τ_xy")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 14)
    principal_diff_MPa = principal_diff / 1e6  # Convert to MPa
    max_diff = np.nanmax(np.abs(principal_diff_MPa))
    plt.pcolormesh(
        X,
        Y,
        principal_diff_MPa,
        cmap="plasma",
        norm=LogNorm(vmax=max_diff, vmin=1e-4 * max_diff),
    )
    plt.colorbar(label="σ₁ - σ₂ (MPa)", shrink=0.8)
    plt.title("Principal Stress\nDifference")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 15)
    max_retardation = np.nanmax(np.abs(retardation))
    plt.pcolormesh(
        X,
        Y,
        retardation,
        cmap="plasma",
        norm=LogNorm(vmin=1e-4 * max_retardation, vmax=max_retardation),
    )
    plt.colorbar(label="Retardation", shrink=0.8)
    plt.title("Retardation")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    # Summary statistics
    plt.subplot(4, 4, 16)
    plt.text(
        0.1,
        0.8,
        f"Load: {P:.0f} N/m",
        fontsize=12,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.7,
        f"Max Fringe Order: {max_fringe:.2f}",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.6,
        f"Max σ₁-σ₂: {max_diff:.2f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.5,
        f"Center σₓₓ: {sigma_xx[n//2, n//2]/1e6:.2f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.4,
        f"Center σᵧᵧ: {sigma_yy[n//2, n//2]/1e6:.2f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.3,
        f"Material f_σ: {f_sigma/1e6:.1f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.2,
        f"Thickness: {t_sample*1000:.0f} mm",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.1,
        f"Wavelength: {lambda_light*1e9:.0f} nm",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.title("Experiment\nParameters")
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.gca().axis("off")

    plt.savefig(outname)


if __name__ == "__main__":
    import os

    import photoelastimetry.io

    # Load the colormap
    # virino_cmap = virino()
    plt.figure(figsize=(12, 12), layout="constrained")

    # Disk and load parameters
    R = 0.0084565       # Radius of the disk (m)
    P = 0.2473/0.00934  # Total load per unit thickness (N/m)

    with open("json/test.json5", "r") as f:
        params = json5.load(f)

    thickness = params["thickness"]  # Thickness in m
    wavelengths_nm = np.array(params["wavelengths"]) * 1e-9  # Wavelengths in nm
    C = np.array(params["C"])  # Stress-optic coefficient (Pa^-1) for each wavelength
    polarisation_efficiency = params["polarisation_efficiency"]  # Polarisation efficiency (0-1)

    # Grid in polar coordinates
    n = 100
    x = np.linspace(-R, R, n)
    y = np.linspace(-R, R, n)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X**2 + Y**2)  # radial distance from center
    mask = R_grid <= R

    cx_px = (n - 1) / 2
    r_px  = (n - 1) / 2
    print(f"Synthetic circle — center: ({cx_px}, {cx_px}) px,  radius: {r_px} px  (image size: {n}×{n})")

    # Get S_i_hat from params if available, otherwise use default
    S_i_hat = np.array(params.get("S_i_hat", [1.0, 0.0, 0.0]))

    # Generate synthetic Brazil test data
    synthetic_images, principal_diff, theta_p, sigma_xx, sigma_yy, tau_xy = generate_synthetic_brazil_test(
        X,
        Y,
        P,
        R,
        S_i_hat,
        mask,
        wavelengths_nm,
        thickness,
        C,
        polarisation_efficiency,
    )

    # Save the output data
    stress = np.stack((sigma_xx, sigma_yy, tau_xy), axis=-1)
    # remove nans
    stress = np.nan_to_num(stress, nan=0.0)

    os.makedirs("images/test", exist_ok=True)

    photoelastimetry.io.save_image("images/test/disk_synthetic_stress.tiff", stress)
    photoelastimetry.io.save_image("images/test/disk_synthetic_images.tiff", synthetic_images)

    fig = plt.figure(figsize=(6, 4), layout="constrained")
    plt.imshow(principal_diff, norm=LogNorm())
    plt.colorbar(label="Principal Stress Difference (Pa)", orientation="vertical")
    plt.savefig("true_stress_difference.png")

    # Post-process and visualize the synthetic data
    for i, lambda_light in enumerate(wavelengths_nm):
        post_process_synthetic_data(
            principal_diff,
            theta_p,
            sigma_xx,
            sigma_yy,
            tau_xy,
            S_i_hat,
            thickness,
            C[i],
            lambda_light,
            f"brazil_test_post_processed_{P:07.0f}_{i:02d}.png",
        )