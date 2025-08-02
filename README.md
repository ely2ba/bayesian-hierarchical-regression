---

# Bayesian Hierarchical Linear Regression (In Progress)

This project implements Bayesian linear regression with **conjugate priors** via a fully custom **Gibbs sampler**, then extends the model with **hierarchical priors** to capture **group-level variation**.

Everything is built from first principles in Python using **only NumPy**—no probabilistic-programming frameworks (e.g., PyMC3, Stan). The result is a transparent, educational walkthrough of Bayesian modeling.

---

## Objectives

* Derive full conditional posteriors for Bayesian linear regression with Normal–Inverse-Gamma priors
* Implement a Gibbs sampler to estimate

  * Regression coefficients (\$\boldsymbol{\beta}\$)
  * Group-level intercepts (\$\gamma\_j\$)
  * Noise variance (\$\sigma^2\$)
  * Group variance (\$\tau^2\$)
* Extend the model to a hierarchical structure for grouped data
* Visualize posterior distributions and shrinkage behaviour
* Compare Bayesian estimates with frequentist OLS
* Explore robustness to prior hyperparameter choices

---

## Features

* Works with both **synthetic** and **real-world datasets** (e.g., Boston Housing)
* Custom implementations of

  * Bayesian linear regression with conjugate priors
  * Gibbs sampler with full-conditional updates
  * Hierarchical structure with partial pooling

### Model Structure

The model treats observations as grouped by a factor (for example, schools or regions):

![Model Equation](https://quicklatex.com/cache3/e9/ql_6ece938cff729788d23a02794467d5e9_l3.png)

Where

* \$\boldsymbol{\beta}\$ – global regression coefficients
* \$\gamma\_{g(i)}\$ – group-specific intercept for observation \$i\$
* \$\sigma^2\$ – observation-level noise variance

Priors

* \$\boldsymbol{\beta} \mid \sigma^2 \sim \mathcal{N}(\boldsymbol{\mu}\_0,; \sigma^2 \Lambda\_0^{-1})\$
* \$\gamma\_j \mid \tau^2 \sim \mathcal{N}(0,; \tau^2)\$
* \$\displaystyle \sigma^2 ;\sim; \operatorname{Inv}\text{-}\Gamma!\bigl(\alpha_0,; \beta_0\bigr)$
* \$\displaystyle \tau^2 ;\sim; \operatorname{Inv}\text{-}\Gamma!\bigl(\alpha_\tau,; \beta_\tau\bigr)$

---

## Diagnostics

The codebase provides

* Trace plots for all sampled parameters
* Posterior histograms and credible intervals
* Comparison with OLS estimates
* Autocorrelation diagnostics (planned)
* Posterior predictive checks (in progress)

---

## Project Structure (coming soon)

```
bayesian-hierarchical-regression/
├── inference/         Gibbs sampler and hierarchical update logic
├── notebooks/         Exploratory analysis and visualisations
├── visuals/           Output plots (posterior distributions, shrinkage, etc.) — coming soon
├── data/              Sample datasets (synthetic + real) — coming soon
└── README.md          Project overview
```

---

## Status

* Core sampler implemented
* Diagnostics and visualisation tools in progress
* Model extensions (random slopes, predictive checks) planned

---

## License

MIT License — use or adapt freely for research or learning.

---

## Planned Next Steps

* Implement posterior predictive simulation and model checking
* Extend to random slopes, allowing \$\boldsymbol{\beta}\_j \sim \mathcal{N}(\boldsymbol{\mu},,\Sigma)\$
* Add support for custom priors and hyperparameter sensitivity analyses
* Optimise sampling performance for large-scale data

---
