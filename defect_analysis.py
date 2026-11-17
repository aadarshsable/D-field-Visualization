import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import definitions as defs
import hm_dfield as hmdf


WIDTHS = [
    ("pi/64", np.pi / 64),
    ("pi/32", np.pi / 32),
    ("pi/16", np.pi / 16),
    ("pi/8",  np.pi / 8),
    ("pi/4",  np.pi / 4),
    ("pi/2",  np.pi / 2),
    ("pi",    np.pi),
]

def get_defect_positions_and_charges(q):
    """Extract (y, x) coordinates and charges s = +/-1 from the winding grid q."""
    ys, xs = np.nonzero(q)
    charges = q[ys, xs]
    coords = np.column_stack((xs, ys)).astype(float)
    return coords, charges

def classify_isolated_defects(coords, charges, L, isolation_radius):
    """
    Use a periodic KD-Tree to find the nearest-neighbor distance (d_nn) for each defect.
    A defect is 'isolated' if d_nn > isolation_radius (e.g., 3 stripe wavelengths).
    """
    if len(coords) < 2:
        return np.array([]), np.ones(len(coords), dtype=bool)
    tree = cKDTree(coords, boxsize=[L, L])
    dists, idx = tree.query(coords, k=2)  # k=1 is self (dist=0), k=2 is nearest neighbor
    d_nn = dists[:, 1]
    is_isolated = d_nn > isolation_radius
    return d_nn, is_isolated

def nested_square_scaling(q, box_sizes, n_origins=64, seed=0):
    """
    Professor's nested-squares method:
    For growing square side lengths l, measure the variance of:
      - Total defect count N(l) = sum |q| inside the l x l square
      - Net topological charge Q(l) = sum q inside the l x l square
    Uses summed-area tables (integral images) for instant exact evaluation.
    """
    L = q.shape[0]
    rng = np.random.default_rng(seed)
    origins = rng.integers(0, L, size=(n_origins, 2))

    mean_N, var_N, var_Q = [], [], []
    for l in box_sizes:
        Ns, Qs = [], []
        for (ox, oy) in origins:
            # Periodic slice of size l x l starting at corner (ox, oy)
            idx_x = (ox + np.arange(l)) % L
            idx_y = (oy + np.arange(l)) % L
            sub = q[np.ix_(idx_y, idx_x)]
            Ns.append(np.sum(np.abs(sub)))
            Qs.append(np.sum(sub))
        mean_N.append(np.mean(Ns))
        var_N.append(np.var(Ns))
        var_Q.append(np.var(Qs))
    return np.array(mean_N), np.array(var_N), np.array(var_Q)

def defect_structure_factor_radial(q, KX, KY, n_bins=80):
    """
    Compute the Number Structure Factor S_N(k) and Charge Structure Factor S_Q(k)
    of the defect positions directly via 2D FFT of the winding charge grid q.
    """
    N_tot = np.sum(np.abs(q))
    if N_tot == 0:
        return None, None, None, None

    # 2D Structure factors: S_Q weights by sign (+/-1), S_N treats all defects as +1
    S_Q_2d = (np.abs(np.fft.fft2(q))**2) / N_tot
    S_N_2d = (np.abs(np.fft.fft2(np.abs(q)))**2) / N_tot
    S_N_2d[0, 0] = 0.0  # Remove the trivial k=0 delta peak (N_tot)

    Kmag = np.hypot(KX, KY)
    k_max = np.pi / 2  # Up to half Nyquist where defect core structure lives
    bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])

    idx = np.digitize(Kmag.ravel(), bins)
    S_Q_rad = np.array([S_Q_2d.ravel()[idx == b].mean() if np.any(idx == b) else np.nan
                        for b in range(1, n_bins + 1)])
    S_N_rad = np.array([S_N_2d.ravel()[idx == b].mean() if np.any(idx == b) else np.nan
                        for b in range(1, n_bins + 1)])

    return k_centers, S_N_rad, S_Q_rad, np.fft.fftshift(S_Q_2d)

def run_all_analyses(alpha=100.0, H=0.0, K_ratio=0.8, angle=0.0, L=2048, sigma=6.0, seed=0):
    X, Y = hmdf.real_grid(L)
    KX, KY = hmdf.freq_grid(L)
    G_hat = hmdf.fft_gaussian_kernel(L, sigma)

    kmag = hmdf.vector_magnitude(L, alpha, H, K_ratio, mode="mean")
    k1, _ = hmdf.perpendicular_basis(kmag, angle, L)
    wavelength = 2 * np.pi / kmag  # ~ 6.1 pixels
    isolation_radius = 3.0 * wavelength  # Isolated if nearest neighbor > 3 wavelengths away (~18 px)

    results = []
    print(f"Stripe wavelength lambda_0 = {wavelength:.2f} px | Isolation threshold = {isolation_radius:.1f} px\n")
    print(f"{'Width':8s} | {'Total Defects':>14s} | {'Isolated (>3 lam)':>18s} | {'Isolated %':>10s} | {'Mean d_nn (px)':>14s}")
    print("-" * 76)

    for label, w in WIDTHS:
        np.random.seed(seed)
        ps, phi = defs.generate_grf(alpha=alpha, H=H, K_ratio=K_ratio, L=L,
                                    angle=angle, angle_width=w / 2)
        eta1 = hmdf.demodulate(phi, k1, G_hat, X, Y)
        q1 = hmdf.winding_number(eta1)

        coords, charges = get_defect_positions_and_charges(q1)
        d_nn, is_iso = classify_isolated_defects(coords, charges, L, isolation_radius)

        n_tot = len(charges)
        n_iso = int(np.sum(is_iso))
        iso_pct = 100.0 * n_iso / n_tot if n_tot > 0 else 0.0
        mean_dnn = np.mean(d_nn) if len(d_nn) > 0 else 0.0

        print(f"{label:8s} | {n_tot:14d} | {n_iso:18d} | {iso_pct:9.1f}% | {mean_dnn:14.2f}")

        results.append(dict(
            label=label, width=w, phi=phi, eta1=eta1, q1=q1,
            coords=coords, charges=charges, d_nn=d_nn, is_iso=is_iso,
            n_tot=n_tot, n_iso=n_iso, iso_pct=iso_pct
        ))

    # -------------------------------------------------------------------------
    # PLOT 1: Defect Distribution & Isolated Defects vs Spectral Width
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    labels = [r["label"] for r in results]
    totals = [r["n_tot"] for r in results]
    isolated = [r["n_iso"] for r in results]
    iso_pcts = [r["iso_pct"] for r in results]

    x_idx = np.arange(len(labels))
    axs[0].plot(x_idx, totals, "o-", lw=2, label="Total Defects ($N_+ + N_-$)")
    axs[0].plot(x_idx, isolated, "s--", lw=2, label=f"Isolated Defects ($d_{{nn}} > 3\\lambda_0$)")
    axs[0].set_xticks(x_idx); axs[0].set_xticklabels(labels)
    axs[0].set_xlabel("Spectral Width $w$ (Narrowest $\\rightarrow$ Isotropic)")
    axs[0].set_ylabel("Defect Count in $2048^2$ Grid")
    axs[0].set_title("1A. Defect Count vs. Spectral Width")
    axs[0].legend(); axs[0].grid(True, alpha=0.3)

    axs[1].bar(x_idx, iso_pcts, color="teal", alpha=0.8)
    axs[1].set_xticks(x_idx); axs[1].set_xticklabels(labels)
    axs[1].set_xlabel("Spectral Width $w$")
    axs[1].set_ylabel("Isolated Defects (%)")
    axs[1].set_title("1B. Fraction of Unbound / Isolated Defects")
    axs[1].grid(True, axis="y", alpha=0.3)

    # Nearest-neighbor distance distribution for narrowest vs isotropic
    for r in [results[0], results[2], results[-1]]:
        if len(r["d_nn"]) > 0:
            axs[2].hist(r["d_nn"] / wavelength, bins=np.linspace(0, 12, 40),
                        density=True, histtype="step", lw=2, label=f"w = {r['label']}")
    axs[2].axvline(3.0, color="red", ls="--", label="Isolation threshold ($3\\lambda_0$)")
    axs[2].set_xlabel("Nearest-Neighbor Distance $d_{nn} / \\lambda_0$ (in wavelengths)")
    axs[2].set_ylabel("Probability Density")
    axs[2].set_title("1C. Nearest-Neighbor Distance Distribution")
    axs[2].legend(); axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("1_defect_distribution_and_isolated.png", dpi=150)

    # -------------------------------------------------------------------------
    # PLOT 2: Visualising Isolated vs Bound Defects in Narrowest (pi/64)
    # -------------------------------------------------------------------------
    r_narrow = results[0]  # pi/64
    crop = 512
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    pc = r_narrow["phi"][:crop, :crop]
    ax.imshow(pc, cmap="gray", vmin=-2*pc.std(), vmax=2*pc.std(), origin="lower")

    in_crop = (r_narrow["coords"][:, 0] < crop) & (r_narrow["coords"][:, 1] < crop)
    c_crop = r_narrow["coords"][in_crop]
    q_crop = r_narrow["charges"][in_crop]
    iso_crop = r_narrow["is_iso"][in_crop]

    # Plot bound dipole defects (small circles) and isolated defects (large stars)
    for is_iso_flag, marker, size, desc in [(False, "o", 50, "Bound Dipole"), (True, "*", 220, "Isolated")]:
        mask_p = (iso_crop == is_iso_flag) & (q_crop > 0)
        mask_m = (iso_crop == is_iso_flag) & (q_crop < 0)
        ax.scatter(c_crop[mask_p, 0], c_crop[mask_p, 1], c="red", marker=marker, s=size,
                   edgecolors="black", label=f"+1 ({desc})")
        ax.scatter(c_crop[mask_m, 0], c_crop[mask_m, 1], c="cyan", marker=marker, s=size,
                   edgecolors="black", label=f"-1 ({desc})")

    ax.set_title(f"Narrowest Spectrum ($w = \\pi/64$): Isolated vs. Bound Defects\n"
                 f"(Stars = Isolated defects with $d_{{nn}} > 3\\lambda_0$)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("2_isolated_defects_visual_pi64.png", dpi=150)

    # -------------------------------------------------------------------------
    # PLOT 3: Professor's Nested Squares (Sub-box Number & Charge Variance)
    # -------------------------------------------------------------------------
    box_sizes = np.unique(np.logspace(np.log10(16), np.log10(512), 12).astype(int))
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

    # 3A: Draw the nested squares with a common corner on top of the defect map!
    ax = axs[0]
    r_demo = results[2]  # pi/16
    demo_crop = 512
    ax.imshow(r_demo["phi"][:demo_crop, :demo_crop], cmap="gray", alpha=0.5, origin="lower")
    in_c = (r_demo["coords"][:, 0] < demo_crop) & (r_demo["coords"][:, 1] < demo_crop)
    ax.scatter(r_demo["coords"][in_c, 0], r_demo["coords"][in_c, 1],
               c=np.where(r_demo["charges"][in_c] > 0, "red", "blue"), s=12)
    for l_sq, col in [(64, "lime"), (128, "yellow"), (256, "orange"), (400, "magenta")]:
        rect = plt.Rectangle((20, 20), l_sq, l_sq, fill=False, edgecolor=col, lw=2.5, label=f"$\\ell = {l_sq}$ px")
        ax.add_patch(rect)
    ax.set_xlim(0, demo_crop); ax.set_ylim(0, demo_crop)
    ax.set_title("3A. Nested Squares with Common Corner\n(Sub-box Sampling of Defects)")
    ax.legend(loc="upper right")

    # 3B & 3C: Variance of Defect Number N(l) and Net Charge Q(l)
    for r in [results[0], results[2], results[4], results[-1]]:
        mean_N, var_N, var_Q = nested_square_scaling(r["q1"], box_sizes)
        axs[1].loglog(box_sizes, var_N, "o-", lw=2, label=f"w = {r['label']}")
        axs[2].loglog(box_sizes, var_Q, "s-", lw=2, label=f"w = {r['label']}")

    # Reference scaling lines: Area (l^2) vs Perimeter (l^1)
    axs[1].loglog(box_sizes, 0.002 * box_sizes**2, "k--", label="$\\propto \\ell^2$ (Poisson / Area)")
    axs[1].set_xlabel("Square Side Length $\\ell$ (px)")
    axs[1].set_ylabel("Number Variance $\\mathrm{Var}[N(\\ell)]$")
    axs[1].set_title("3B. Defect Number Variance ($\\propto \\ell^2$)")
    axs[1].legend(); axs[1].grid(True, which="both", alpha=0.3)

    axs[2].loglog(box_sizes, 0.02 * box_sizes**1, "k--", label="$\\propto \\ell^1$ (Perimeter / Screening)")
    axs[2].loglog(box_sizes, 0.0005 * box_sizes**2, "k:", label="$\\propto \\ell^2$ (Uncorrelated)")
    axs[2].set_xlabel("Square Side Length $\\ell$ (px)")
    axs[2].set_ylabel("Net Charge Variance $\\mathrm{Var}[Q(\\ell)]$")
    axs[2].set_title("3C. Charge Variance (Hyperuniform $\\propto \\ell^1$)")
    axs[2].legend(); axs[2].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("3_nested_squares_scaling.png", dpi=150)

    # -------------------------------------------------------------------------
    # PLOT 4: Structure Factor of Defect Positions S_N(k) and S_Q(k)
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for r in [results[0], results[2], results[4], results[-1]]:
        k_c, S_N_rad, S_Q_rad, _ = defect_structure_factor_radial(r["q1"], KX, KY)
        if k_c is not None:
            axs[0].plot(k_c, S_N_rad, lw=2, label=f"w = {r['label']}")
            axs[1].plot(k_c, S_Q_rad, lw=2, label=f"w = {r['label']}")

    axs[0].axhline(1.0, color="k", ls="--", label="Ideal Gas ($S(k)=1$)")
    axs[0].set_xlabel("Wave number $k$ (rad/px)")
    axs[0].set_ylabel("Number Structure Factor $S_N(k)$")
    axs[0].set_title("4A. Defect Position Structure Factor $S_N(k)$")
    axs[0].legend(); axs[0].grid(True, alpha=0.3)

    axs[1].axhline(1.0, color="k", ls="--", label="Uncorrelated ($S(k)=1$)")
    axs[1].set_xlabel("Wave number $k$ (rad/px)")
    axs[1].set_ylabel("Charge Structure Factor $S_Q(k)$")
    axs[1].set_title("4B. Defect Charge Structure Factor $S_Q(k) \\to 0$ as $k \\to 0$")
    axs[1].legend(); axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("4_defect_structure_factor.png", dpi=150)
    print("\nSaved all 4 figures successfully!")

if __name__ == "__main__":
    run_all_analyses()