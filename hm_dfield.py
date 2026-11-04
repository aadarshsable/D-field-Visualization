import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import definitions as defs


def real_grid(L):
    x = np.arange(L)
    return np.meshgrid(x, x)

def freq_grid(L):
    k = np.fft.fftfreq(L) * 2 * np.pi
    return np.meshgrid(k, k)

def fft_gaussian_kernel(L, sigma):
    """
    Returns the FFT of a normalised Gaussian kernel.
    Normalised Gaussian function is G(n) = 1/(sqrt(2 pi) sigma) exp(-|n|^2 / (2 sigma^2))
    FFT[G](m) numerically equals to exp(-sigma^2 |m|^2 / 2).
    """
    x = np.arange(L)
    d = np.minimum(x, L - x)
    DX, DY = np.meshgrid(d, d)
    G = np.exp(-(DX**2 + DY**2) / (2 * sigma**2))
    G /= G.sum()
    return np.real(np.fft.fft2(G))


def vector_magnitude(L, alpha, H, K_ratio, mode="mean"):
    """
    returns magnitude used for our perpendicular basis wave vectors.
      'peak' : K = K_ratio * f_max (inner edge of the ring, where psi_hat = 1 first)
      'mean' : mean |k| of the radial spectrum psi_hat (centre of the band)
    """
    f_max = np.pi / 3
    K_val = K_ratio * f_max
    if mode == "peak":
        return K_val
    KX, KY = freq_grid(L)
    Kmag = np.hypot(KX, KY)
    p = defs.psi_hat(Kmag, alpha, H, K_val, f_max)
    return np.sum(Kmag * p) / np.sum(p)


def snap_to_grid(k, L):
    """Snap a wave vector to the nearest FFT grid point 2 pi n / L to keep exp(-ik.x) periodic."""
    dk = 2 * np.pi / L
    return np.round(np.asarray(k) / dk) * dk


def perpendicular_basis(kmag, angle, L):
    """Return two perpendicular wave vectors with magnitude kmag and angle relative to the x-axis."""
    k1 = snap_to_grid([kmag * np.cos(angle), kmag * np.sin(angle)], L)
    k2 = snap_to_grid([-kmag * np.sin(angle), kmag * np.cos(angle)], L)
    return k1, k2


def demodulate(phi, k0, G_hat, X, Y):
    """ returns eta, where eta = IFFT{ FFT[phi exp(-i k0.x)] * FFT[G] }  ==  (phi exp(-i k0.x)) (*) G"""
    shifted = phi * np.exp(-1j * (k0[0] * X + k0[1] * Y))
    return np.fft.ifft2(np.fft.fft2(shifted) * G_hat)


def d_field(eta, KX, KY):
    """
    D = d_x eta_R d_y eta_I - d_x eta_I d_y eta_R  =  Im( conj(d_x eta) * d_y eta )
    Derivatives are spectral (exact for periodic fields): d_x f = IFFT[i k_x FFT f].
    """
    eh = np.fft.fft2(eta)
    ex = np.fft.ifft2(1j * KX * eh)
    ey = np.fft.ifft2(1j * KY * eh)
    return ex.real * ey.imag - ex.imag * ey.real


def winding_number(eta):
    """Integer topological charge of each 2x2 kernel: (1/2pi) * sum of wrapped phase jumps."""
    th = np.angle(eta)
    w = lambda x: np.angle(np.exp(1j * x))  # returns the shortest signed angle between 2 phases
    a = th
    b = np.roll(th, -1, 1)  # shifts the array 'th' 1 pixel to the left
    c = np.roll(b, -1, 0)   # shifts the array 'b' 1 pixel down
    d = np.roll(th, -1, 0)  # shifts the array 'th' 1 pixel down

    # rounded to the nearest intger due to floating point errors
    return np.rint((w(b - a) + w(c - b) + w(d - c) + w(a - d)) / (2 * np.pi)).astype(int)


CONFIGS = [
    ("Fully Isotropic", np.pi),
    ("Mild Anisotropy", np.pi / 2),
    ("Moderate Anisotropy", np.pi / 4),
    ("Strong Anisotropy", np.pi / 16),
]


def compare_dfields(alpha=100.0, H=0.0, K_ratio=0.8, angle=0.0, L=2048,
                    sigma=6.0, k_mode="mean", crop=256, seed=0,
                    out="dfield_comparison.png"):
    
    X, Y = real_grid(L)
    KX, KY = freq_grid(L)
    G_hat = fft_gaussian_kernel(L, sigma)

    kmag = vector_magnitude(L, alpha, H, K_ratio, k_mode)
    k1, k2 = perpendicular_basis(kmag, angle, L)

    n = len(CONFIGS)
    fig, axs = plt.subplots(n, 5, figsize=(25, 5 * n))
    stats = []

    for i, (label, width) in enumerate(CONFIGS):
        np.random.seed(seed)        # same random phases in every row means only the anisotropy changes
        ps, phi = defs.generate_grf(alpha=alpha, H=H, K_ratio=K_ratio, L=L,
                                    angle=angle, angle_width=width / 2)

        # demodulating on the entire field but cropping it for display
        eta1 = demodulate(phi, k1, G_hat, X, Y)
        eta2 = demodulate(phi, k2, G_hat, X, Y)
        D1 = d_field(eta1, KX, KY)
        D2 = d_field(eta2, KX, KY)

        # common normalisation per row (rho = D / (pi eta0^2))
        eta0_sq = 0.5 * (np.mean(np.abs(eta1)**2) + np.mean(np.abs(eta2)**2))
        r1 = D1 / (np.pi * eta0_sq)
        r2 = D2 / (np.pi * eta0_sq)

        q1, q2 = winding_number(eta1), winding_number(eta2)
        s = dict(label=label,
                 eta1_rms=np.sqrt(np.mean(np.abs(eta1)**2)),
                 eta2_rms=np.sqrt(np.mean(np.abs(eta2)**2)),
                 D1_mean=D1.mean(), D2_mean=D2.mean(),
                 n1p=(q1 > 0).sum(), n1m=(q1 < 0).sum(),
                 n2p=(q2 > 0).sum(), n2m=(q2 < 0).sum())
        stats.append(s)

        # takes the 99.5th percentile of the pixels to account for a few skewed pixels
        vmax = np.percentile(np.abs(np.concatenate([r1[:crop, :crop].ravel(), r2[:crop, :crop].ravel()])), 99.5)

        # column 1: spectrum with annotated k1 and k2
        ps_vis = maximum_filter(np.log10(np.fft.fftshift(ps) / (ps.max() + 1e-12) + 1e-12), size=3)
        c = L // 2
        dk = 2 * np.pi / L
        ax = axs[i, 0]
        ax.imshow(ps_vis, cmap="hot", vmin=-4, vmax=0, origin="lower")
        for kn, col, name in [(k1, "cyan", r"$\mathbf{k}_1$"), (k2, "lime", r"$\mathbf{k}_2$")]:
            ax.annotate("", xy=(c + kn[0] / dk, c + kn[1] / dk), xytext=(c, c),
                        arrowprops=dict(arrowstyle="->", color=col, lw=2))
            ax.add_patch(plt.Circle((c + kn[0] / dk, c + kn[1] / dk), 1 / sigma / dk,
                                    fill=False, color=col, ls="--", lw=1.2))
        ax.set_xlim(c - 600, c + 600); ax.set_ylim(c - 600, c + 600)
        ax.set_title(f"{label}\n" + r"$\widehat{\psi}(\mathbf{k})$" + f" (Width: $\\pi/{np.pi / width:.0f}$)")
        ax.axis("off")

        # column 2: field
        pc = phi[:crop, :crop]
        lim = 2 * pc.std()
        axs[i, 1].imshow(pc, cmap="bwr", vmin=-lim, vmax=lim, origin="lower")
        axs[i, 1].set_title(r"Scalar Field $\varphi(\mathbf{x})$")
        axs[i, 1].axis("off")

        # columns 3, 4: D-fields
        for j, (r, nm) in enumerate([(r1, "1"), (r2, "2")]):
            im = axs[i, 2 + j].imshow(r[:crop, :crop], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            axs[i, 2 + j].set_title(rf"$D(\eta_{nm})\,/\,\pi\Psi_0^2$")
            axs[i, 2 + j].axis("off")
            fig.colorbar(im, ax=axs[i, 2 + j], fraction=0.046, pad=0.04)

        # column 5: D contours on top of the field
        ax = axs[i, 4]
        ax.imshow(pc, cmap="bwr", vmin=-lim, vmax=lim, origin="lower")
        t = 0.5 * vmax
        ax.contour(r1[:crop, :crop], levels=[-t, t], colors=["yellow", "black"], linewidths=1.2)
        ax.contour(r2[:crop, :crop], levels=[-t, t], colors=["cyan", "lime"], linewidths=1.2)
        ax.set_title(r"$D_1=\pm t$ (yellow/black), $D_2=\pm t$ (cyan/green)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"saved {out}")

    print(f"\n{'case':22s} {'rms|eta1|':>10s} {'rms|eta2|':>10s} {'<D1>':>10s} {'<D2>':>10s}"
          f" {'N1+/N1-':>12s} {'N2+/N2-':>12s}")
    for s in stats:
        n2 = "n/a" if s["eta2_rms"] < 1e-3 * s["eta1_rms"] else f"{s['n2p']}/{s['n2m']}"
        print(f"{s['label']:22s} {s['eta1_rms']:10.4f} {s['eta2_rms']:10.1e} {s['D1_mean']:10.1e} "
              f"{s['D2_mean']:10.1e} {s['n1p']:>5d}/{s['n1m']:<6d} {n2:>12s}")
    return stats


if __name__ == "__main__":
    compare_dfields(alpha=100.0, H=0.0, K_ratio=0.8, angle=0.0, sigma=6.0)