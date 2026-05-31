import numpy as np
from photoelastimetry.image import compute_normalised_stokes, compute_stokes_components

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
    Process photoelastic image data up to and including Step 2.
    Returns wrapped retardance and principal stress orientation.

    Parameters
    ----------
    data : ndarray
        Raw image data with shape (H, W, n_wavelengths, 4),
        where the last axis is polariser angles [0, 45, 90, 135].
    S_i_hat : array-like, optional
        Incoming normalised Stokes vector. Defaults to linear horizontal [1, 0, 0].

    Returns
    -------
    theta : ndarray, shape (H, W)
        Principal stress orientation in radians.
    delta_wrap : ndarray, shape (H, W, n_wavelengths)
        Wrapped retardance in radians [0, pi].
    """
    H, W = data.shape[:2]

    # Step 1 — Stokes extraction
    I_0   = data[..., 0]
    I_45  = data[..., 1]
    I_90  = data[..., 2]
    I_135 = data[..., 3]

    S0, S1, S2 = compute_stokes_components(I_0, I_45, I_90, I_135)
    S1_hat, S2_hat = compute_normalised_stokes(S0, S1, S2)
    S_m_hat = np.stack([S1_hat, S2_hat], axis=-1)

    S_flat = S_m_hat.reshape(-1, S_m_hat.shape[-2], S_m_hat.shape[-1])

    # Step 2 — Invert to wrapped retardance and orientation
    if S_i_hat is None:
        S_i_hat = np.array([1.0, 0.0, 0.0])
    S_i_hat = _normalise_input_stokes_vector(S_i_hat)

    theta, delta_wrap = invert_wrapped_retardance(S_flat, S_i_hat)

    theta = theta.reshape(H, W)
    delta_wrap = delta_wrap.reshape(H, W, -1)

    return theta, delta_wrap