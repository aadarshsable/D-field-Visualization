# D-fields from Demodulation of Anisotropic GRFs

## Prelude

We generate a real, periodic field on an $L\times L$ grid:

$$
\varphi(\mathbf{x}) = \frac{1}{L^2}\sum_{\mathbf{k}} \hat\varphi(\mathbf{k})\, e^{i\mathbf{k}\cdot\mathbf{x}},
\qquad \hat\varphi(\mathbf{k}) = \nu\,F_h(\mathbf{k}),\qquad \hat\varphi(-\mathbf{k}) = \hat\varphi(\mathbf{k})^*,
$$

where $F_h$ is `F_hermitian` and $\mathbf{k} = 2\pi\mathbf{n}/L$. The amplitude picture on the slides writes the same field as

$$
\varphi(\mathbf{x}) = \sum_j \Big[\eta_j(\mathbf{x})\, e^{i\mathbf{k}_j\cdot\mathbf{x}} + \eta_j^*(\mathbf{x})\, e^{-i\mathbf{k}_j\cdot\mathbf{x}}\Big],
$$

with slowly varying complex amplitudes $\eta_j$. The goal is to extract $\eta_j$ for a chosen $\mathbf{k}_j$.

## Demodulation Theory

Let us fix a wave vector $\mathbf{k}_{j_0}$.

**Step 1:** We multiply the field $\varphi(\mathbf{x})$ by the conjugate carrier to isolate fast oscillating components

```math
\psi_{j_0}(\mathbf{x}) = \varphi(\mathbf{x})\, e^{-i\mathbf{k}_{j_0}\cdot\mathbf{x}}
= \eta_{j_0} + \eta_{j_0}^*\, e^{-2i\mathbf{k}_{j_0}\cdot\mathbf{x}}
+ \sum_{j\neq j_0}\Big[\eta_j\, e^{i(\mathbf{k}_j-\mathbf{k}_{j_0})\cdot\mathbf{x}} + \eta_j^*\, e^{-i(\mathbf{k}_j+\mathbf{k}_{j_0})\cdot\mathbf{x}}\Big]
```

Only $\eta_{j_0}$ is left without a fast oscillation.

**Step 2:** We convolve with the normalised 2D Gaussian ($G(\mathbf{r})$) to eliminate the fast oscillating component

$$
G(\mathbf{r}) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{|\mathbf{r}|^2}{2\sigma^2}\right)
$$

then the amplitude is defined as

$$
\eta_{j_0}(\mathbf{x}) = (\psi_{j_0} * G)(\mathbf{x})
$$

**Step 3:** We then fourier transform $\eta_{j_0}$ to convert the convolution to multiplication,

$$
\mathcal{F}[\eta_{j_0}](\mathbf{q}) = \mathcal{F}[\psi_{j_0}](\mathbf{q})\cdot \mathcal{F}[G](\mathbf{q})
$$

The Fourier transform of a Gaussian is again a Gaussian:

$$
\mathcal{F}[G](\mathbf{q}) = \int G(\mathbf{r})\, e^{-i\mathbf{q}\cdot\mathbf{r}}\, d^2r = \exp\!\left(-\frac{\sigma^2 |\mathbf{q}|^2}{2}\right)
$$

The real space obversational window width $\sigma$ becomes a Fourier space width $1/\sigma$, and $\mathcal{F}[G](0) = 1$ because $G$ is normalised.

The $k_{j_0}$ radian phase shift in step 1 is a $k_{j_0}$ translation in Fourier space, $\mathcal{F}[\psi_{j_0}](\mathbf{q}) = \hat\varphi(\mathbf{k}_{j_0} + \mathbf{q})$, so

$$
{\;\hat\eta_{j_0}(\mathbf{q}) = \hat\varphi(\mathbf{k}_{j_0}+\mathbf{q})\; e^{-\sigma^2|\mathbf{q}|^2/2}\;}
$$

**Step 4:** We now inverse fourier transform $\hat\eta_{j_0}$ to get $\eta_{j_0}$

$$
\eta_{j_0}(\mathbf{x}) = \mathcal{F}^{-1}\Big\{\mathcal{F}\big[\varphi\, e^{-i\mathbf{k}_{j_0}\cdot\mathbf{x}}\big]\cdot e^{-\sigma^2|\mathbf{q}|^2/2}\Big\}
= \frac{1}{L^2}\sum_{\mathbf{q}} \hat\varphi(\mathbf{k}_{j_0}+\mathbf{q})\, e^{-\sigma^2|\mathbf{q}|^2/2}\, e^{i\mathbf{q}\cdot\mathbf{x}}
$$

where $L$ is the grid size of our physical field $\varphi(\mathbf{x})$

So $\eta_{j_0}$ is the summation of the Fourier transforms of $\varphi$ within roughly $1/\sigma$ observational window of $\mathbf{k}_{j_0}$, each multiplied by the Gaussian.


## 3. Perpendicular wave vector "bases"

Now, we want to take two perpendicular wave vectors ($k_1$ and $k_2$) and demodulate the field $\varphi$ along these. Take

$$
\mathbf{k}_1 = k_0(\cos\theta_0,\ \sin\theta_0), \qquad
\mathbf{k}_2 = k_0(-\sin\theta_0,\ \cos\theta_0)
$$

where we can take $\theta_0$ = 0, so $\mathbf{k}_1$ lies along the remaining spectrum (in the fourier space) and $\mathbf{k}_2$ is perpendicular to it. And, $k_0$ is taken as the weighted mean radius of the spectral band. In our case, we get it as

$$
k_0 = \frac{\sum_{\mathbf{k}} |\mathbf{k}|\,\hat\psi(|\mathbf{k}|)}{\sum_{\mathbf{k}} \hat\psi(|\mathbf{k}|)} \approx 1.029 \ \text{rad/px},
$$

which is the centre of the band.

$k_1$ and $k_2$ are then snapped to the FFT grid, since FFT requres our $L \times L$ grid/field to be periodic, so we need a whole number of wavelength cycles in the grid. 

$$
L = n \cdot \frac{2\pi}{k}
$$

where $n$ is a (whole) number or waves in the grid and our target wave vector $k = k_0$. In our case we get $n=335.4$, and snapping the waves to the grid gives us $n=335$.

We can now compute each $\eta_j$ independently though demodulation of the same field $\varphi$:

$$
\eta_1 = \mathcal{F}^{-1}\!\big\{\mathcal{F}[\varphi e^{-i\mathbf{k}_1\cdot\mathbf{x}}]\,\mathcal{F}[G]\big\},\qquad
\eta_2 = \mathcal{F}^{-1}\!\big\{\mathcal{F}[\varphi e^{-i\mathbf{k}_2\cdot\mathbf{x}}]\,\mathcal{F}[G]\big\}.
$$

## 4. D-field computation from the Halperin-Mazenko method

According to the Halperin-Mazenko method, we write $\eta_j = \eta_{j,R} + i\,\eta_{j,I}$. Then we do a coordinate transformation $(x, y) \mapsto (\eta_{j,R}, \eta_{j,I})$. This maps each physical point in the $L \times L$ grid to a point in an abstract "order parameter space".

A topological defect occurs when $\eta = 0$, i.e. $\eta_R = \eta_I = 0$. So theoretically, we need to look at the origin(s) in the order parameter space and reverse the coordinate transformation to get the exact point in the physical space where the defect is.

Since we do not have a neat mathematical function for this coordinate transformation, we look at the Jacobian determinant of this transformation:

$$
D_j = \frac{\partial(\eta_{j,R},\eta_{j,I})}{\partial(x,y)} = \partial_x\eta_{j,R}\,\partial_y\eta_{j,I} - \partial_x\eta_{j,I}\,\partial_y\eta_{j,R}
$$

**Singular defect density:**

$$
\rho_{{top}_j}(\mathbf{r}) = D_j\,\delta^{(2)}(\eta_j) = \sum_a q_a\,\delta^{(2)}(\mathbf{r}-\mathbf{r}_a)
$$
with $s_a = \operatorname{sign} D_j(\mathbf{r}_a) = \pm 1$ at each zero $\mathbf{r}_a$ of $\eta_j$.

This gives us a space that is zero everywhere but blows up to infinity at the exact point of the topological defects.

Since we can't plot this we "smear" this infinite spike by normalizing the function.

**Non-singular defect density:**

 $$\rho_j = D_j / (\pi\eta_0^2)$$
 
 For plotting we use this quantity. I set $\eta_0^2 = \tfrac12\big(\langle|\eta_1|^2\rangle + \langle|\eta_2|^2\rangle\big)$ per row, so $D_1$ and $D_2$ have a common normalizing factor.


## 5. Visualisation

In `dfield_comparison.png` we have rows with spectral width $\pi$, $\pi/2$, $\pi/4$, $\pi/16$, and the same random phases (seed) in every row. The columns are:

1. $\hat\psi(\mathbf{k})$ with $\mathbf{k}_1$ (cyan) and $\mathbf{k}_2$ (green), and dashed circles of radius $1/\sigma$ showing the demodulation observation windows.
2. $\varphi(\mathbf{x})$.
3. $D(\eta_1)/\pi\eta_0^2$.
4. $D(\eta_2)/\pi\eta_0^2$.
5. $\varphi$ with contours at $\rho = \pm t$, $t = 0.5\max|\rho|$.

The fields are computed on the full $2048 \times 2048$ periodic grid and cropped to $256 \times 256$ for ease of display.

## 6. Results ($\sigma = 6$ px, seed 0, $L = 2048$)

| Anisotropy Angle | rms $\lvert\eta_1\rvert$ | rms $\lvert\eta_2\rvert$ | defects in $\eta_1$ ($N_+/N_-$) | defects in $\eta_2$ |
|---|---|---|---|---|
| Fully isotropic ($\pi$) | 0.0330 | 0.0333 | 3663 / 3663 | 3567 / 3567 |
| Mild ($\pi/2$) | 0.0452 | $6.5\times10^{-7}$ | 3663 / 3663 | none (no power) |
| Moderate ($\pi/4$) | 0.0620 | $2.3\times10^{-11}$ | 3614 / 3614 | none (no power) |
| Strong ($\pi/16$) | 0.1215 | $7.8\times10^{-15}$ | 1695 / 1695 | none (no power) |

## 7. Interpretation

$D_1$ barely changes until the spectral width is narrower than the demodulation observational window. The window around $\mathbf{k}_1$ has angular width of about $1/(\sigma k_0) \approx 9^\circ$. Spectral widths of $90^\circ$, $45^\circ$ and $22.5^\circ$ contain it entirely. Because the phases are the same, $\eta_1$ is the same field up to the global factor $\nu$, which gives identical defect counts for $\pi$ and $\pi/2$. 

At $\pi/16$ the observational window spills out of the spectral band. $\eta_1$ then becomes more coherent (larger $|\eta_1|$), and the number of dislocations halves. This is consistent with the nearly perfect stripes in that row.

Hence, for a fixed demodulation window $\sigma$, the defect count in $D_1$ only drops when the anisotropy angle becomes narrower than the window's angular width $1/\sigma k_0$. Conversely, to faithfully capture an entire anisotropic spectral band without artificial clipping, the observational window must be wider than the anisotropy angle.