# This is the same code from the previous task, except I have changed //
# the code for generate_grf to allow for anisotropy in the GRFs

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.ndimage import maximum_filter
import gudhi
import skfmm

def enforce_hermitian(F):
    """Return Hermitian-symmetric spectrum so ifft2 result is real."""
    L = F.shape[0]
    # Get the indices for the conjugate pairs: F[i, j] and F[-i, -j]
    i = np.arange(L)
    j = np.arange(L)
    ineg = (-i) % L
    jneg = (-j) % L
    
    # Average F with its flipped, conjugated counterpart
    F_conj_pair = np.conj(F[np.ix_(ineg, jneg)])
    Fh = 0.5 * (F + F_conj_pair)
    return Fh

def psi_hat(k, alpha, H, K, f_max):
    """Spectral density function as defined in Eq 3.2"""
    psi = np.zeros_like(k)
    
    m1 = k <= K
    if K > 0:
        psi[m1] = (k[m1] / K)**alpha * (1 - H) + H
    else:
        psi[m1] = H # To prevent division by 0
        
    m2 = (k > K) & (k <= 0.8 * f_max)
    psi[m2] = 1.0
    
    m3 = k > 0.8 * f_max
    psi[m3] = np.exp(-10 * (0.8 * f_max - k[m3])**2)
    
    return psi

def generate_grf(alpha, H, K_ratio, K_tilde_ratio=None, L=2048, angle=0.0, angle_width=np.pi):
    """
    Generates the GRF scalar field as described in Eq 3.1
    The argument 'angle' controls the direction of anisotropy
    The argument 'angle_width' controls the range of the angle over which the anisotropy is applied //
    (e.g. if angle_width=pi/4, then the anisotropy is applied to angles in [angle - pi/8, angle + pi/8])
    """
    # Create frequency grid
    kx = np.fft.fftfreq(L) * 2 * np.pi
    ky = np.fft.fftfreq(L) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_mag = np.sqrt(KX**2 + KY**2)
    
    # f_max = (100 * np.pi) / L # As described in the paper
    # Above value did not work
    f_max = np.pi / 3 # This makes the shortest wavelength ~6 pixels, as Abel explained to me
    K_val = K_ratio * f_max # As defined in the paper
    
    # Base HU Spectrum
    psi_dense = psi_hat(K_mag, alpha, H, K_val, f_max)
    
    # Spectrum tweak for stealthy Non-HU
    if K_tilde_ratio is not None:
        K_tilde = K_tilde_ratio * f_max
        psi_dense[K_mag <= K_tilde] = 1.0

    # Calculate angle of each k vector
    theta = np.arctan2(KY, KX) # Range [-pi, pi]

    # Applying anisotropy mask
    if angle_width < np.pi:
        # We want to keep angles within [angle - width, angle + width]
        # We should also keep the opposite direction [angle + pi] to //
        # ensure the final real-space image is a real and not complex field.

        # Calculate angular difference to our fixed angle
        diff1 = np.abs(np.arctan2(np.sin(theta - angle), np.cos(theta - angle))) # Range [0, pi]

        # Calculate angular difference to opposite direction
        diff2 = np.abs(np.arctan2(np.sin(theta - angle - np.pi), np.cos(theta - angle - np.pi))) # Range [0, pi]

        # Create a mask that is 1 where either diff1 or diff2 is within the angle_width, and 0 elsewhere
        mask = np.logical_or(diff1 <= angle_width, diff2 <= angle_width)

        psi_dense *= mask.astype(float) # Zero out frequencies outside the desired angle range

    # Add random phases
    phases = np.random.uniform(0, 2*np.pi, (L, L))
    F_complex = np.sqrt(psi_dense) * np.exp(1j * phases) # Amplitude (A in the paper) is sqrt(psi_hat)

    # Enforce Hermitian symmetry for real-valued field
    F_hermitian = enforce_hermitian(F_complex)
    F_hermitian[0, 0] = 0.0 # Ensure zero mean (no DC offset)
    
    spectral_density = np.abs(F_hermitian)**2
    
    # Inverse FFT to get the physical scalar field
    phi = np.real(np.fft.ifft2(F_hermitian)) # The code for the paper did not use ifft; maybe need to modify
    
    # Normalize field so that max|phi(x)| = 1
    nu = 1.0 / np.max(np.abs(phi))
    phi = nu * phi
    
    return spectral_density, phi

def compute_signed_distance(phi, c=0.0):
    """Signed distance as defined in Eq 2.7"""
    # mask_gt = phi > c
    # mask_lt = phi <= c
    
    # # EDT measures distance to '0' elements. We want distance to the boundary.
    # dist_to_boundary_gt = ndimage.distance_transform_edt(mask_gt)
    # dist_to_boundary_lt = ndimage.distance_transform_edt(mask_lt)
    
    # signed_dist = np.zeros_like(phi)
    # signed_dist[mask_gt] = -dist_to_boundary_gt[mask_gt]
    # signed_dist[mask_lt] = dist_to_boundary_lt[mask_lt]
    
    # Above comment did not work, so;
    # Using skfmm to compute signed distance directly
    signed_dist = skfmm.distance(c - phi)

    return signed_dist

def compute_ph(signed_dist):
    """Computes Persistent Homology using Gudhi"""
    cc = gudhi.CubicalComplex(
        dimensions=signed_dist.shape, 
        top_dimensional_cells=signed_dist.flatten()
    )
    cc.compute_persistence()
    p0 = cc.persistence_intervals_in_dimension(0)
    p1 = cc.persistence_intervals_in_dimension(1)
    return p0, p1

def plot_row(alpha, H, K_ratio, K_tilde_ratio=None):
    # Generate field
    ps, phi = generate_grf(alpha, H, K_ratio, K_tilde_ratio=K_tilde_ratio)
    
    # Crop to 256x256 as stated in the paper
    phi_crop = phi[:256, :256]
    
    # Persistent Homology
    signed_dist = compute_signed_distance(phi)
    p0, p1 = compute_ph(signed_dist)
    
    # Visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Spectral Density
    ps_shifted = np.fft.fftshift(ps)
    ps_norm = ps_shifted / (np.max(ps_shifted) + 1e-12)
    ps_log = np.log10(ps_norm + 1e-12) # log scale as used in the code for the paper
    ps_vis = maximum_filter(ps_log, size=3)

    axs[0].imshow(ps_vis, cmap='hot', vmin=-4.0, vmax=0.0) # vmin and vmax values as used in the code for the paper
    center = ps_vis.shape[0] // 2
    axs[0].set_xlim(center - 600, center + 600)
    axs[0].set_ylim(center - 600, center + 600) # Zoom into the central region
    axs[0].set_title(r'$\widehat{\psi}(\mathbf{k})$')
    axs[0].axis('off')
    
    # Scalar Field phi(x)
    phi_std = np.std(phi_crop)
    phi_lim = max(2.0 * phi_std, 1e-6)
    im = axs[1].imshow(phi_crop, cmap='bwr', vmin=-phi_lim, vmax=phi_lim)
    axs[1].set_title(r'$\varphi(\mathbf{x})$')
    axs[1].axis('off')
    
    # Persistence Diagram
    axs[2].scatter(p0[:, 0], p0[:, 1], c='blue', s=5, alpha=0.5, label=r'$\mathcal{P}_0$')
    if len(p1) > 0:
        axs[2].scatter(p1[:, 0], p1[:, 1], c='red', s=5, alpha=0.5, label=r'$\mathcal{P}_1$')
    
    axs[2].plot([-10, 10], [-10, 10], c='gray', lw=1)
    axs[2].fill_between([-10, 10], [-10, 10], -10, color='lightgray', alpha=0.5) # death < birth region
    axs[2].set_xlim(-10, 10)
    axs[2].set_ylim(-10, 10)
    axs[2].set_xlabel('birth $r$')
    axs[2].set_ylabel('death $r$')
    axs[2].legend(loc='lower right')
    
    plt.tight_layout()