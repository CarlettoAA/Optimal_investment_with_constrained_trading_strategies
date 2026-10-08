# Numerical Solution of Constrained BSDEs for Option Pricing under Non-Convex Portfolio Constraints

**Penalisation, Least-Squares Monte Carlo and Physics-Informed Neural Networks**

Reference implementation accompanying the Master's thesis *Optimal Investments with Constrained Trading Strategies* by *Carlo Alberto Angelini*, *Technische Universität München, Chair of Mathematical Finance*, *2025*.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Framework](#2-mathematical-framework)
3. [Repository Structure](#3-repository-structure)
4. [Method A: Picard Iteration with Least-Squares Monte Carlo](#4-method-a-picard-iteration-with-least-squares-monte-carlo-lsmc)
5. [Method B: Physics-Informed Neural Networks](#5-method-b-physics-informed-neural-networks-pinn)
6. [Prerequisites](#6-prerequisites)
7. [Installation and Usage](#7-installation-and-usage)
8. [Parameters and Conventions](#8-parameters-and-conventions)
9. [Validation and Benchmarks](#9-validation-and-benchmarks)
10. [Reproducibility Notes](#10-reproducibility-notes)
11. [References](#11-references)
12. [Citation and License](#12-citation-and-license)

---

## 1. Overview

This repository contains the numerical experiments of the thesis. The common problem is the pricing of a European call option on a dividend-paying stock when the hedging portfolio is subject to **constraints on the admissible strategies**, including **non-convex** constraints. In the BSDE (backward stochastic differential equation) approach to super-replication, such constraints are enforced by **penalising** the generator of a linear BSDE, so that the price is obtained as the limit of a sequence of penalised, non-linear BSDEs.

Two independent numerical approaches are implemented and compared:

| | Method A: LSMC | Method B: PINN |
|---|---|---|
| **Idea** | Picard iteration of a forward-type scheme, with conditional expectations approximated by least-squares regression on an orthonormalised basis | A neural network approximates the pricing function $`u(t,x)`$ and is trained on a time-discretised BSDE residual along simulated trajectories |
| **Unknowns** | $`(Y_{t_j}, Z_{t_j})`$ on simulated paths | $`u_\theta(t,x)`$ on the whole space-time domain |
| **Framework** | NumPy / SciPy | TensorFlow 2 (nested automatic differentiation) |
| **Location** | [`lsmc/`](lsmc) | [`pinn/`](pinn) |

Both methods are validated against the closed-form Black–Scholes price with continuous dividend yield (the unconstrained benchmark).

---

## 2. Mathematical Framework

### 2.1 Market model

Under the physical measure $`\mathbb P`$, a stock with continuous dividend yield $`\delta`$ and a riskless bond with rate $`r`$:

```math
dS_t = S_t\,(\mu\,dt + \sigma\,dW_t), \qquad dB_t = rB_t\,dt .
```

The claim is a European call on the dividend-adjusted stock:

```math
\xi = G(S_T) = \big(S_T e^{-\delta T} - K\big)^+ .
```

### 2.2 Hedging BSDE

Let $`\pi_t`$ denote the amount of wealth invested in the stock. The wealth process $`Y`$ solves

```math
-dY_t = f(Y_t, Z_t)\,dt - Z_t\,dW_t, \qquad Y_T = \xi,
\qquad f(y,z) = r y + \frac{\mu - r}{\sigma}\, z, \qquad Z_t = \sigma \pi_t .
```

Pricing is performed under $`\mathbb P`$; the market price of risk $`\theta = (\mu-r)/\sigma`$ appears in the generator (equivalently, one changes to $`\mathbb Q`$ via Girsanov's theorem).

### 2.3 Constraints and penalisation

Constraints of the form $`\pi_t \in \mathcal C(Y_t)`$ (no short-selling, no borrowing, or a non-convex borrowing constraint of the type $`\pi_t - Y_t \le f_{\mathrm{con}}(Y_t)`$) are enforced by the penalised generator

```math
f_\lambda(y,z) = r y + \frac{\mu - r}{\sigma}\, z \;-\; \lambda\; \mathrm{pen}(y,z),
```

with $`\lambda \ge 0`$ the penalisation weight, and, for instance,

```math
\mathrm{pen}(y,z) = \Big(-\tfrac{z}{\sigma}\Big)^+ \quad\text{(no short-selling)},\qquad
\mathrm{pen}(y,z) = \Big(\tfrac{z}{\sigma} - y - f_{\mathrm{con}}(y)\Big)^+ \quad\text{(non-convex borrowing)} .
```

The non-convex borrowing function $`f_{\mathrm{con}}`$ is piecewise defined by the concavity parameter $`u`$, the exponent $`q`$ and the threshold $`\rho`$ (`f_con` in the code). As $`\lambda \to \infty`$ the penalised solutions $`Y^\lambda`$ are expected to approximate the minimal super-replication price under the constraint. The scripts therefore compute $`Y_0^\lambda`$ for a **sweep of weights** and study the convergence in $`\lambda`$.

The PINN scripts additionally implement a **non-convex integrality-type penalty**, namely the distance of the hedge ratio $`\partial_x u`$ to the nearest integer (`f_frac`, `f_frac_N`).

---

## 3. Repository Structure

```
constrained-bsde-numerics/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── lsmc/                                   # Method A: Picard + least-squares Monte Carlo
│   ├── penalized_bsde_lsmc.py              # Main solver, original scale (S0 = 100, K = 110)
│   ├── penalized_bsde_lsmc_normalized.py   # Same solver, normalised scale (S0 = 1, K = 1.1)
│   └── lsmc_projection_demo.py             # Stand-alone demo of the L2 projection used by LSMC
│
└── pinn/                                   # Method B: physics-informed neural networks
    ├── pinn_baseline.py                    # Plain network ansatz u = NN(t, x)
    ├── pinn_additive.py                    # Hard terminal condition: u = (T - t) NN + g(x)
    └── pinn_exponential.py                 # Hard terminal condition: u = exp((T - t) NN) g(x)
```

### File-by-file description

| File | Purpose |
|---|---|
| `lsmc/penalized_bsde_lsmc.py` | Penalised BSDE solved by Picard iterations; 100 000 paths, 5 time points, weights $`\lambda = 0,\dots,10`$; the active penalty is the no-short-selling term $`\frac{\sqrt\lambda}{100}\,(-z/\sigma)^+`$. Includes the convergence test on $`\mathbb E[Y_0]`$, a check of the sign of $`Z`$, and a plot of $`Y_0`$ against $`\lambda`$. |
| `lsmc/penalized_bsde_lsmc_normalized.py` | Variant on the normalised scale ($`S_0=1`$, $`K=1.1`$), 1 000 paths, 10 time points, weights $`\lambda=0,\dots,30`$, penalty $`\frac{\lambda}{10}(-z/\sigma)^+`$. Contains commented alternatives for other payoffs (put, straddle, smoothed call) and other penalties. |
| `lsmc/lsmc_projection_demo.py` | Pedagogical script: projects a noisy target $`Y=\sin X+\varepsilon`$ onto polynomial bases $`\{1,X,X^2\}`$ and $`\{1,\dots,X^4\}`$ via least squares, to illustrate the $`L^2`$-projection that underlies the LSMC step. |
| `pinn/pinn_baseline.py` | PINN with unconstrained network output; smoothed (softplus) terminal condition enforced through a terminal loss. |
| `pinn/pinn_additive.py` | Terminal condition built into the ansatz additively; integer-distance penalty added to the drift of the residual. |
| `pinn/pinn_exponential.py` | Terminal condition built into the ansatz multiplicatively (exponential parametrisation, which keeps $`u>0`$). |

---

## 4. Method A: Picard Iteration with Least-Squares Monte Carlo (LSMC)

### 4.1 Scheme

On a grid $`0=t_0<\dots<t_{N-1}=T`$ with step $`\Delta_j=t_{j+1}-t_j`$, a **forward-type scheme** (in the spirit of Bender & Denk) is iterated over $`k`$:

```math
Y^{k}_{t_j} = \mathbb E\Big[\xi - \sum_{i=j}^{N-2} f_\lambda\big(Y^{k-1}_{t_i}, Z^{k-1}_{t_i}\big)\Delta_i \;\Big|\; \mathcal F_{t_j}\Big],
```

```math
Z^{k}_{t_j} = \mathbb E\Big[\Big(\xi - \sum_{i=j+1}^{N-2} f_\lambda\big(Y^{k-1}_{t_i}, Z^{k-1}_{t_i}\big)\Delta_i\Big)\frac{\Delta W_j}{\Delta_j} \;\Big|\; \mathcal F_{t_j}\Big].
```

Each Picard step involves only *linear* regressions, because the (non-linear) generator is evaluated at the previous iterate. The iteration for a given $`\lambda`$ stops when $`\big|\overline{Y^{k}_0}-\overline{Y^{k-1}_0}\big|\le 10^{-4}`$ (and, in the original-scale script, only once no negative $`Z`$ values remain).

### 4.2 Conditional expectations

Conditional expectations are replaced by $`L^2(\mathbb P_L)`$-projections onto the span of basis functions evaluated at $`S_{t_j}`$:

```math
\{\,1,\; x-S_0,\; (x-S_0)^2,\; (x-S_0)^3,\; G(x)\,\}.
```

The terminal payoff is included in the basis to capture the kink of the call. At each time step the design matrix $`\Phi`$ is **orthonormalised w.r.t. the empirical scalar product** through a thin SVD, $`\Phi = U\Sigma V^\top`$, and columns with singular values below $`10^{-10}`$ are discarded (numerical rank truncation). With $`\Psi=\sqrt L\,U`$ one has $`\tfrac1L\Psi^\top\Psi=I`$ and the regression coefficients reduce to $`\beta=\tfrac1L\Psi^\top b`$.

---

## 5. Method B: Physics-Informed Neural Networks (PINN)

### 5.1 Principle

A fully connected network (4 hidden layers, 256 neurons, `tanh` activations) is trained, over $`10^5`$ simulated GBM trajectories, to satisfy a time-discretised BSDE relation between consecutive grid points $`t_i \to t_{i+1}`$. The residual $`\mathcal R_i`$ involves $`u_\theta`$, its first and second spatial derivatives (obtained by **nested `tf.GradientTape`**) and the Brownian increment $`\Delta W_i`$; the loss is

```math
\mathcal L(\theta) = h^{\,p/2-1}\;\mathbb E\big|u_\theta(T,X_T) - g(X_T)\big|^{p} + \sum_{i=0}^{N-1}\mathbb E\,|\mathcal R_i|^{p},\qquad p=2 .
```

The terminal payoff is replaced by a smooth softplus approximation, $`g_\varepsilon(x)=\varepsilon\log\!\big(1+e^{(xe^{-\delta T}-K)/\varepsilon}\big)`$ with $`\varepsilon=0.01`$, so that $`g`$ is differentiable.

### 5.2 Network ansätze

| Script | Ansatz for $`u_\theta(t,x)`$ | Terminal condition | Integer-distance penalty |
|---|---|---|---|
| `pinn_baseline.py` | $`\mathrm{NN}_\theta(t,x)`$ | soft (terminal loss) | not in residual |
| `pinn_additive.py` | $`(T-t)\,\mathrm{NN}_\theta(t,x)+g_\varepsilon(x)`$ | exact by construction | added to the drift of the residual |
| `pinn_exponential.py` | $`\exp\!\big((T-t)\,\mathrm{NN}_\theta(t,x)\big)\,g_\varepsilon(x)`$ | exact by construction | tracked only (diagnostic) |

### 5.3 Training

Adam optimiser, exponential learning-rate decay (initial rate $`10^{-4}`$, factor $`0.7`$ every 3 000 steps), 25 000 epochs, mini-batches of 400 trajectories drawn from the simulated set, `float32` arithmetic.

---

## 6. Prerequisites

### 6.1 Mathematical background

- **Stochastic calculus**: Brownian motion, Itô's formula, geometric Brownian motion, Girsanov's theorem and change of measure.
- **Mathematical finance**: the Black–Scholes model, replication and super-replication, hedging under portfolio constraints, dividends.
- **BSDE theory**: existence and uniqueness for Lipschitz generators, linear BSDEs and their connection to option pricing, comparison theorems, the penalisation approach to constrained BSDEs, and the links between BSDEs and semilinear PDEs.
- **Numerical analysis of BSDEs**: time discretisation, Picard iteration, regression-based (LSMC / Gobet–Lemor–Warin) approximation of conditional expectations, forward schemes.
- **Linear algebra**: SVD, orthogonal projections, least squares.
- **Optimisation and deep learning**: stochastic gradient methods (Adam), automatic differentiation, physics-informed neural networks, issues of stiffness and loss balancing.

### 6.2 Software

- Python ≥ 3.9
- NumPy, SciPy, Matplotlib
- TensorFlow ≥ 2.10 (PINN scripts only). A GPU is strongly recommended: each PINN training involves 25 000 steps with second-order derivatives over all time steps.

---

## 7. Installation and Usage

```bash
git clone https://github.com/CarlettoAA/Optimal_investment_with_constrained_trading_strategies.git
cd constrained-bsde-numerics

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Each script is self-contained and is run directly:

```bash
# Method A: LSMC
python lsmc/penalized_bsde_lsmc.py
python lsmc/penalized_bsde_lsmc_normalized.py
python lsmc/lsmc_projection_demo.py

# Method B: PINN
python pinn/pinn_baseline.py
python pinn/pinn_additive.py
python pinn/pinn_exponential.py
```

Scripts print progress and results to the console and open Matplotlib windows (`plt.show()`), so a graphical backend is required; on a headless machine set `MPLBACKEND=Agg` and replace `plt.show()` by `plt.savefig(...)`.

**Output of the LSMC solvers**: $`Y_0^\lambda`$ for each weight $`\lambda`$ of the sweep, the number of Picard iterations to convergence, and (original-scale script) a plot of $`Y_0`$ against the penalisation weight.

**Output of the PINN scripts**: loss histories (total and per time step), the learned $`u_\theta(t,\cdot)`$ at $`t=0`$ and at intermediate times compared with Black–Scholes, 2D and 3D surfaces of the solution, absolute-error maps, and mean squared / absolute errors on a test grid and on freshly simulated paths.

---

## 8. Parameters and Conventions

| Symbol | Code | Value | Meaning |
|---|---|---|---|
| $`r`$ | `r` | 0.05 | Risk-free rate |
| $`\mu`$ | `mu` | 0.07 | Stock drift under $`\mathbb P`$ |
| $`\sigma`$ | `sigma` | 0.2 | Volatility |
| $`\delta`$ | `delta` | 0.1 | Continuous dividend yield |
| $`T`$ | `T` | 0.5 | Maturity |
| $`S_0`$ | `S0` | 100 (original) / 1 (normalised) | Initial stock price |
| $`K`$ | `K` | 110 (original) / 1.1 (normalised) | Strike |
| $`N`$ | `N`, `N_steps` | 5 / 10 | Number of time points / steps |
| $`L`$ | `L`, `N_paths` | $`10^5`$ (original LSMC), $`10^3`$ (normalised LSMC), $`10^5`$ (PINN) | Simulated paths |
| $`\lambda`$ | `weight` / `num_weight` | sweep | Penalisation weight |
| $`u,q,\rho`$ | `u`/`concav`, `q`, `rho` | 9, variable, 1000 | Parameters of the non-convex borrowing function |

**Normalisation.** Several scripts work with $`S_0=1`$ and $`K=1.1`$ for numerical conditioning, i.e. all monetary quantities are divided by 100. Results are mapped back via $`U(t,S)=100\,u^*(t,S/100)`$ (see the "Scaled" blocks of the PINN scripts).

**Time grids.** In the LSMC scripts `N` is the number of *grid points* and the step is $`T/(N-1)`$; in the PINN scripts `N_steps` is the number of *intervals* and the step is $`T/N_{\text{steps}}`$.

---

## 9. Validation and Benchmarks

The unconstrained price of the call with dividend yield is available in closed form,

```math
C(t,S)=S e^{-\delta(T-t)}\Phi(d_1)-Ke^{-r(T-t)}\Phi(d_2),\qquad
d_{1,2}=\frac{\ln(S/K)+(r-\delta\pm\tfrac12\sigma^2)(T-t)}{\sigma\sqrt{T-t}},
```

and is implemented as `black_scholes_call_div`. It serves as reference for:

- the value at $`\lambda=0`$ in the LSMC scripts (unpenalised BSDE);
- the PINN solution on the whole domain (pointwise errors, MSE on a test grid and along simulated trajectories);
- an **empirical $`\mathbb P`$-measure benchmark** (`price_call_at_given_time`), a Monte Carlo estimator that prices under $`\mathbb P`$ by multiplying the discounted payoff with the Girsanov density $`\exp(-\theta\,\Delta W-\tfrac12\theta^2\tau)`$.

---

## 10. Reproducibility Notes

- Random seeds are set at the top of the scripts (`np.random.seed`, `tf.random.set_seed`). In `pinn_additive.py` the seed lines are commented out; uncomment them for deterministic runs.
- GPU non-determinism in TensorFlow can still cause small run-to-run differences.
- Many hyperparameters (grid sizes, number of epochs, penalty variants) are module-level constants and several alternative penalty/payoff definitions are retained as commented-out code, documenting the experiments carried out in the thesis.
- Everything is `float32` in the PINN scripts; in the neighbourhood of the strike the softplus smoothing parameter $`\varepsilon`$ and the floating-point precision interact with the integer-distance penalty, which operates on the sixth decimal place.

---

## 11. References

1. C. Bender and J. Denk, *A forward scheme for backward SDEs*, Stochastic Processes and their Applications 117 (2007).
2. C. Bender and M. Kohlmann, *Optimal superhedging under nonconvex constraints: a BSDE approach*, 2008. *(Example 5.2 of this work is the model case implemented in the LSMC scripts; please verify the bibliographic details.)*
3. N. El Karoui, S. Peng and M.-C. Quenez, *Backward stochastic differential equations in finance*, Mathematical Finance 7 (1997).
4. J. Cvitanić and I. Karatzas, *Hedging contingent claims with constrained portfolios*, Annals of Applied Probability 3 (1993).
5. E. Gobet, J.-P. Lemor and X. Warin, *A regression-based Monte Carlo method to solve backward stochastic differential equations*, Annals of Applied Probability 15 (2005).
6. F. Longstaff and E. Schwartz, *Valuing American options by simulation: a simple least-squares approach*, Review of Financial Studies 14 (2001).
7. M. Raissi, P. Perdikaris and G. E. Karniadakis, *Physics-informed neural networks*, Journal of Computational Physics 378 (2019).

---

## 12. Citation and License

If you use this code in academic work, please cite the thesis:

```bibtex
@mastersthesis{carloalbertoangelini_thesis,
  author  = {Carlo Alberto Angelini},
  title   = {Optimal Investments with Constrained Trading Strategies},
  school  = {Technische Universität München},
  year    = {2025}
}
```

License: *MIT License*.
