# 📊 Bayesian Hierarchical Linear Regression (In Progress)

This project implements Bayesian linear regression with **conjugate priors** using a fully custom **Gibbs sampler**, and extends the model to incorporate **hierarchical priors** that account for **group-level variation**.

The entire workflow is implemented from first principles in Python using **only NumPy**. No probabilistic programming libraries (e.g., PyMC3, Stan) are used. This makes the project a transparent and educational tool for understanding Bayesian modeling.

---

## 📚 Objectives

- Derive full conditional posteriors for Bayesian linear regression with Normal-Inverse-Gamma priors  
- Implement a Gibbs sampler to estimate:
  - Regression coefficients ($\boldsymbol{\beta}$)
  - Group-level intercepts ($\gamma_j$)
  - Noise variance ($\sigma^2$)
  - Group variance ($\tau^2$)
- Extend the model to a hierarchical structure for grouped data
- Visualize posterior distributions and shrinkage behavior
- Compare Bayesian estimates with frequentist OLS
- Explore robustness to prior hyperparameter choices

---

## 🔧 Features

- Works with both **synthetic** and **real-world datasets** (e.g., Boston Housing)
- Custom implementation of:
  - Bayesian linear regression with conjugate priors
  - Gibbs sampler with full conditional updates
  - Hierarchical structure with partial pooling

### 📐 Model Structure

The model assumes observations grouped by some factor (e.g., schools, regions):

$$
y_i = \mathbf{x}_i^\top \boldsymbol{\beta} + \gamma_{g(i)} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

Where:
- $\boldsymbol{\beta}$: global regression coefficients
- $\gamma_{g(i)}$: group-specific intercept for observation $i$
- $\sigma^2$: observation-level noise variance

Priors:
- $\boldsymbol{\beta} \mid \sigma^2 \sim \mathcal{N}(\boldsymbol{\mu}_0, \sigma^2 \Lambda_0^{-1})$
- $\gamma_j \mid \tau^2 \sim \mathcal{N}(0, \tau^2)$
- $\sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)$
- $\tau^2 \sim \text{Inv-Gamma}(\alpha_\tau, \beta_\tau)$

---

## 📈 Diagnostics

Includes tools for:

- Trace plots of all sampled parameters
- Posterior histograms and credible intervals
- Comparison with OLS estimates
- Autocorrelation diagnostics (coming soon)
- Posterior predictive checks (in progress)

---
## 🗂 Project Structure (coming soon)

```
bayesian-hierarchical-regression/
├── inference/         Gibbs sampler and hierarchical update logic
├── notebooks/         Exploratory analysis and visualizations
├── visuals/           Coming soon : Output plots (posterior distributions, shrinkage, etc.)
├── data/              Coming soon : Sample datasets (synthetic + real) 
└── README.md          Project overview
```
---

## 🚧 Status

- ✔️ Core sampler implemented
- 🔄 Diagnostics and visualization tools in progress
- 🔬 Model extensions planned (e.g., random slopes, predictive checks)

---

## 📜 License

MIT License — feel free to use or adapt for your own research or learning.

---

## 💡 Planned Next Steps

- Implement posterior predictive simulation and model checking
- Extend to random slopes: allow $\boldsymbol{\beta}_j \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$
- Add support for custom priors and hyperparameter sensitivity
- Optimize sampling performance for large-scale data
