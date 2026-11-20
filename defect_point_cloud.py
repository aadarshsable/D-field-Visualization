import numpy as np
import matplotlib.pyplot as plt
import definitions as defs
import hm_dfield as hm

L = 2048
sigma = 6.0
X, Y = hm.real_grid(L)
KX, KY = hm.freq_grid(L)
G_hat = hm.fft_gaussian_kernel(L, sigma)
kmag = hm.vector_magnitude(L, alpha=100.0, H=0.0, K_ratio=0.8, mode="mean")
k1, k2 = hm.perpendicular_basis(kmag, angle=0.0, L=L)

widths = [("pi/32", np.pi/32),
          ("pi/64", np.pi/64),
          ("pi/128", np.pi/128),
          ("pi/256", np.pi/256)]

fig, axs = plt.subplots(len(widths), 3, figsize=(16, 4 * len(widths)))

for i, (label, w) in enumerate(widths):
    np.random.seed(0)
    ps, phi = defs.generate_grf(alpha=100.0, H=0.0, K_ratio=0.8, L=L, angle=0.0, angle_width=w/2)
    eta1 = hm.demodulate(phi, k1, G_hat, X, Y)
    D1 = hm.d_field(eta1, KX, KY)
    q1 = hm.winding_number(eta1)

    ys, xs = np.nonzero(q1)
    charges = q1[ys, xs]

    # Raw D1 full 2048*2048 image
    psi0 = np.sqrt(np.mean(np.abs(eta1)**2))
    rho_raw = D1 / (np.pi * psi0**2)
    vmax = np.percentile(np.abs(rho_raw), 99.5)
    axs[i, 0].imshow(rho_raw, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    axs[i, 0].set_title(f"{label}: Raw $D_1/\\pi\\Psi_0^2$")
    axs[i, 0].axis("off")

    # Normalized HM density: D1 * exp(-|eta1|^2 / eps^2)
    eps = 0.25 * psi0
    rho_norm = D1 * np.exp(-np.abs(eta1)**2 / eps**2) / (np.pi * eps**2)
    vloc = np.percentile(np.abs(rho_norm), 99.9)
    axs[i, 1].imshow(rho_norm, cmap="RdBu_r", vmin=-vloc, vmax=vloc, origin="lower")
    axs[i, 1].set_title(f"{label}: Normalized $D_1 \\cdot \\delta_\\epsilon(\\eta_1)$")
    axs[i, 1].axis("off")

    # Defect Point Cloud
    axs[i, 2].scatter(xs[charges > 0], ys[charges > 0], c="red", s=12, label=f"+1 ({(charges>0).sum()})")
    axs[i, 2].scatter(xs[charges < 0], ys[charges < 0], c="blue", s=12, label=f"-1 ({(charges<0).sum()})")
    axs[i, 2].set_xlim(0, L); axs[i, 2].set_ylim(0, L)
    axs[i, 2].set_aspect("equal")
    axs[i, 2].set_title(f"{label}: Defect Point Cloud ($2048^2$ full image)\nTotal = {len(charges)} defects")
    axs[i, 2].legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("pointcloud.png", dpi=150)
print("Saved pointcloud.png")