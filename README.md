# EECS 6699 Final Project: The Expressive Power of Depth — and Its Robustness Under Noise

**Columbia University · Department of Electrical Engineering**
**Course:** EECSE 6699 — *Mathematics of Deep Learning* (Prof. Predrag R. Jelenković)
**Author:** Yiwen Chen · Spring 2026

---

## 1. Project Overview

This project investigates the **mathematical foundations of depth in neural networks** along two tightly coupled axes:

1. **Expressivity** — *Does depth provide an exponential representational advantage over width?*
2. **Robustness** — *Does that advantage survive when the input or the supervision signal is corrupted by noise?*

Building on the *Depth Separation Theorem* of **Telgarsky (2016)**, we first reproduce the classical result that a deep, narrow ReLU network can fit a high-frequency sawtooth function that any parameter-matched shallow network provably cannot. We then **extend the baseline** with a systematic empirical study of **noise robustness**, asking whether the deep network's exponential expressivity is a fragile artifact of clean data or a structural property that persists under adversarial and stochastic perturbations.

The compositional, piecewise-linear nature of ReLU produces $2^L$ linear regions with depth $L$ but only $O(W)$ regions with width $W$. Whether those exponentially many regions are *useful* or merely *brittle* under noise is the central empirical question of the extended study.

---

## 2. Research Questions

| # | Question | Type |
| :-- | :-- | :-- |
| **RQ1** | Under a fixed parameter budget, does a deep narrow network strictly outperform a parameter-matched shallow wide network on iterated sawtooth targets? | Reproduction |
| **RQ2** | When inputs are corrupted by Gaussian noise $x \mapsto x + \epsilon,\ \epsilon \sim \mathcal{N}(0,\sigma^2)$, does the depth advantage persist, shrink, or invert? | Extension |
| **RQ3** | When training labels are corrupted by Gaussian noise $y \mapsto y + \delta$, do deep networks overfit the noise faster than shallow ones (consistent with their higher expressivity)? | Extension |
| **RQ4** | Under bounded-$\ell_\infty$ adversarial perturbations (FGSM / PGD), is there a **critical perturbation budget** $\epsilon^\star$ at which the deep model's advantage collapses? | Extension |
| **RQ5** | How does the **empirical Lipschitz constant** of the trained network scale with depth, and does it predict the robustness gap observed in RQ2–RQ4? | Theory ↔ Experiment |

---

## 3. Theoretical Background

### 3.1 The Compositional Power of ReLU

The ReLU activation $\sigma(x) = \max(0, x)$ is piecewise linear, so any ReLU network computes a continuous piecewise-linear function. Define the *mirror operator*
$$\phi(x) \;=\; |2x - 1|,$$
which a 2-layer ReLU network represents exactly with two hidden units. Iterating $\phi$ yields the sawtooth family
$$f_k(x) \;=\; \underbrace{\phi \circ \phi \circ \cdots \circ \phi}_{k\text{ times}}(x), \qquad k \in \mathbb{N}.$$
Each composition **doubles** the number of linear regions, so $f_k$ has exactly $2^k$ peaks on $[0,1]$.

### 3.2 Depth Separation (Telgarsky, 2016)

> **Theorem (informal).** For every $k \in \mathbb{N}$, there exists a function $f_k:[0,1]\to\mathbb{R}$ representable by a ReLU network of depth $O(k)$ and width $O(1)$, such that *any* network of depth 2 must use $\Omega(2^k)$ neurons to approximate $f_k$ to within constant $L^1$ error.

The depth gain is therefore **exponential in $k$**. Width is a strictly weaker resource than depth for this function class.

### 3.3 Why Noise Matters: A Lipschitz Bound

For a piecewise-linear network $g_\theta$, the worst-case sensitivity to input perturbation is governed by its (local) Lipschitz constant
$$L(g_\theta) \;=\; \sup_{x \neq x'} \frac{\| g_\theta(x) - g_\theta(x') \|}{\| x - x' \|}.$$
A classical product bound gives
$$L(g_\theta) \;\le\; \prod_{\ell=1}^{L} \| W_\ell \|_2,$$
which grows **multiplicatively in depth**. This is precisely the source of the central tension in our extension:

- *Telgarsky* tells us depth buys exponential expressivity.
- *Lipschitz analysis* warns us depth may also buy exponential sensitivity to noise.

The empirical study in §5 asks which effect dominates in the regime we can train.

### 3.4 Linear Regions and Robustness

Following **Hanin & Rolnick (2019)**, the *expected* number of linear regions of a randomly initialized ReLU network grows polynomially in width but only modestly with depth at initialization — yet trained networks fitting $f_k$ must approach the worst-case $2^L$ count. We therefore predict a **robustness phase transition**: noise of magnitude $\sigma$ erases linear regions whose width is $\lesssim \sigma$, collapsing the deep network's effective expressivity once $\sigma \gtrsim 2^{-k}$.

---

## 4. Hypotheses

- **H1 (Clean expressivity).** With matched parameter count $N$, the deep network achieves test MSE that is at least an order of magnitude lower than the shallow network on $f_4$.
- **H2 (Input-noise resilience).** For small $\sigma$ (input noise $\sigma \ll 2^{-k}$), the deep network retains its advantage; for $\sigma \gtrsim 2^{-k}$, the gap closes monotonically and may invert.
- **H3 (Label-noise overfitting).** The deep network's training loss decays faster than the shallow network's even on noisy labels, but its *test* loss is higher — consistent with depth memorizing noise.
- **H4 (Adversarial threshold).** There exists a critical $\epsilon^\star \approx 2^{-k}$ above which the deep network's adversarial robust accuracy drops below the shallow network's.
- **H5 (Lipschitz mediation).** The robustness gap in H2–H4 is quantitatively explained by the empirical Lipschitz constant ratio $L(g_{\text{deep}})/L(g_{\text{shallow}})$.

---

## 5. Experimental Design

### 5.1 Target Function and Models

- **Target.** Iterated sawtooth $f_k(x) = \phi^{(k)}(x)$ with $k \in \{2, 3, 4, 5, 6\}$ to study how the depth advantage scales with target complexity.
- **Deep–Narrow Model.** Depth $L = 2k+1$, width $W = 4$ (e.g. $L=9,\ W=4$ for $k=4$).
- **Shallow–Wide Model.** Depth $L = 2$, width $W$ chosen so total parameter counts match within $\pm 2\%$.
- **Optimizer.** Adam, learning rate $3\times10^{-3}$, $1.5\times10^4$ epochs, MSE loss.
- **Seeds.** Five random seeds per configuration; we report mean ± std.

### 5.2 Phase 1 — Baseline (Clean Depth Separation) *(reproduces existing notebook)*

Train both models on clean $(x, f_k(x))$ pairs. Record final MSE, training curves, and qualitative fits. Confirms RQ1 / H1.

### 5.3 Phase 2 — Input Noise Sweep *(extension)*

For $\sigma_x \in \{0,\ 10^{-4},\ 10^{-3},\ 10^{-2},\ 10^{-1}\}$:
1. Train on clean data.
2. Evaluate on $\tilde{x} = x + \mathcal{N}(0, \sigma_x^2)$.
3. Plot **MSE vs. $\sigma_x$** for both models on the same axes.
4. Identify $\sigma_x^\star$ at which the curves cross (the *critical noise level*).

Tests RQ2 / H2.

### 5.4 Phase 3 — Label Noise Sweep *(extension)*

Train on $(x,\ f_k(x) + \mathcal{N}(0, \sigma_y^2))$ for $\sigma_y \in \{0, 10^{-3}, 10^{-2}, 10^{-1}\}$, evaluate on clean targets. Track:
- Train MSE vs. epoch (memorization speed)
- Final clean test MSE (true generalization)
- *Generalization gap* = test MSE − train MSE

Tests RQ3 / H3.

### 5.5 Phase 4 — Adversarial Robustness *(extension)*

For each trained model, attack with **FGSM** and **PGD-20** at $\ell_\infty$ budgets $\epsilon \in \{0,\ 10^{-3},\ 10^{-2},\ 5\!\times\!10^{-2}\}$:
$$x_{\text{adv}} \;=\; \mathrm{Proj}_{[0,1]}\left(x + \epsilon \cdot \mathrm{sign}(\nabla_x \mathcal{L}(g_\theta(x), f_k(x)))\right).$$
Measure adversarial MSE and locate $\epsilon^\star$ per RQ4 / H4.

### 5.6 Phase 5 — Lipschitz and Linear-Region Diagnostics *(extension)*

- **Empirical Lipschitz.** Estimate $\hat{L}$ by maximum finite-difference on a $10^4$-point grid and via the product of operator norms $\prod_\ell \|W_\ell\|_2$.
- **Linear region count.** Following Hanin & Rolnick, sample the activation pattern of the network on a fine grid and count distinct patterns; compare to the $2^L$ upper bound.
- **Cross-validation.** Regress the Phase 2–4 robustness gaps on $\hat{L}$ to test H5.

### 5.7 Phase 6 — Complexity Scaling *(extension)*

Repeat Phases 1–4 for $k \in \{2, 3, 4, 5, 6\}$. We expect $\sigma_x^\star, \epsilon^\star \propto 2^{-k}$, providing a **quantitative law** for the robustness–complexity trade-off.

---

## 6. Key Metrics & Deliverables

| Metric | Phase | Reports On |
| :-- | :-- | :-- |
| Clean test MSE | 1 | Expressivity (H1) |
| Test MSE vs. $\sigma_x$ curve, crossover $\sigma_x^\star$ | 2 | Input robustness (H2) |
| Generalization gap vs. $\sigma_y$ | 3 | Label-noise overfitting (H3) |
| Adversarial MSE, $\epsilon^\star$ | 4 | Adversarial robustness (H4) |
| Empirical Lipschitz $\hat{L}$, linear-region count | 5 | Mechanistic explanation (H5) |
| Scaling law for $\sigma_x^\star,\epsilon^\star$ vs. $k$ | 6 | Complexity-dependent transition |

**Deliverables.** (i) Reproducible Jupyter notebook covering all six phases, (ii) a 6–8 page LaTeX research report, (iii) an oral-defense slide deck summarizing theory, results, and limitations.

---

## 7. Repository Structure

```
EECS6699_Final_Project/
├── README.md                         # This document
├── 6699_final.ipynb                  # Phase 1 baseline (existing)
├── notebooks/
│   ├── phase2_input_noise.ipynb      # to be added
│   ├── phase3_label_noise.ipynb
│   ├── phase4_adversarial.ipynb
│   ├── phase5_lipschitz_regions.ipynb
│   └── phase6_complexity_scaling.ipynb
├── src/
│   ├── models.py                     # ModelBuilder, parameter-matching utilities
│   ├── targets.py                    # sawtooth_target, alternative compositional targets
│   ├── noise.py                      # input/label perturbations, FGSM, PGD
│   ├── diagnostics.py                # Lipschitz estimator, linear-region counter
│   └── train.py                      # training loop with seed control
├── results/
│   ├── figures/                      # all plots produced for the report
│   └── tables/                       # CSV summaries for the LaTeX paper
├── report/
│   ├── paper.tex
│   └── refs.bib
└── slides/
    └── presentation_outline.md
```

---

## 8. Installation & Usage

**Prerequisites.** Python ≥ 3.10; PyTorch ≥ 2.1; NumPy; Matplotlib; SciPy.

```bash
# Clone the repository
git clone https://github.com/Yiwen543/EECS6699_Final_Project.git
cd EECS6699_Final_Project

# Install dependencies
pip install torch numpy matplotlib scipy jupyterlab

# Reproduce the baseline (Phase 1)
jupyter lab 6699_final.ipynb

# Run extension phases (once notebooks are added)
jupyter lab notebooks/phase2_input_noise.ipynb
```

All experiments run on a single CPU in under 30 minutes; a GPU is helpful for Phase 6 only.

---

## 9. Project Timeline

| Week | Milestone |
| :-- | :-- |
| 1 | Phase 1 reproduction; finalize README and hypotheses |
| 2 | Implement `src/noise.py`, `src/diagnostics.py`; Phase 2 sweep |
| 3 | Phase 3 (label noise) and Phase 4 (FGSM/PGD) experiments |
| 4 | Phase 5 Lipschitz / linear-region diagnostics |
| 5 | Phase 6 complexity scaling; draft LaTeX report |
| 6 | Final report and presentation |

---

## 10. References

1. Telgarsky, M. (2016). *Benefits of depth in neural networks.* COLT.
2. Eldan, R., & Shamir, O. (2016). *The power of depth for feedforward neural networks.* COLT.
3. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). *Explaining and harnessing adversarial examples.* ICLR.
4. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards deep learning models resistant to adversarial attacks.* ICLR.
5. Hanin, B., & Rolnick, D. (2019). *Deep ReLU networks have surprisingly few activation patterns.* NeurIPS.
6. Sokolić, J., Giryes, R., Sapiro, G., & Rodrigues, M. R. D. (2017). *Robust large-margin deep neural networks.* IEEE TSP.
7. Bartlett, P. L., Foster, D. J., & Telgarsky, M. (2017). *Spectrally-normalized margin bounds for neural networks.* NeurIPS.
8. Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). *Understanding deep learning requires rethinking generalization.* ICLR.
9. Jelenković, P. R. (2026). *EECS 6699 — Mathematics of Deep Learning, Lecture Notes.* Columbia University.
