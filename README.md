# Bayesian Hierarchical Linear Regression (In Progress)

This project implements Bayesian linear regression using conjugate priors and a custom Gibbs sampler, then extends the model to a hierarchical prior structure to account for group-level variation. Everything is built from first principles in Python—no probabilistic programming libraries (e.g., PyMC3, Stan) are used.

---

## 📚 Objectives

- Derive the full posterior distribution for a linear regression model using Normal-Inverse-Gamma conjugate priors  
- Implement a Gibbs sampler to estimate posterior distributions of coefficients and noise variance  
- Extend the model with hierarchical priors to model multilevel group structures  
- Visualize posterior distributions, predictive uncertainty, and shrinkage effects  
- Compare Bayesian estimates with frequentist OLS and assess robustness to different prior specifications
# Bayesian Hierarchical Linear Regression (In Progress)

This project implements Bayesian linear regression using conjugate priors and a custom Gibbs sampler, then extends the model to a hierarchical prior structure to account for group-level variation. Everything is built from first principles in Python—no probabilistic programming libraries (e.g., PyMC3, Stan) are used.

---

## 📚 Objectives

- Derive the full posterior distribution for a linear regression model using Normal-Inverse-Gamma conjugate priors  
- Implement a Gibbs sampler to estimate posterior distributions of coefficients and noise variance  
- Extend the model with hierarchical priors to model multilevel group structures  
- Visualize posterior distributions, predictive uncertainty, and shrinkage effects  
- Compare Bayesian estimates with frequentist OLS and assess robustness to different prior specifications

---

## 🔧 Features

- Works with both synthetic and real-world datasets (e.g., Boston Housing)  
- Custom implementations of:
  - Closed-form Bayesian posterior updates  
  - Gibbs sampling algorithm  
  - Hierarchical priors of the form:  
    βⱼ ~ N(μⱼ, τ²), μⱼ ~ N(μ₀, σ²)
- Diagnostic tools:
  - Trace plots
  - Posterior predictive checks
  - Autocorrelation plots

---
## 🗂 Project Structure (coming soon)

```
bayesian-hierarchical-regression/
├── inference/         Gibbs sampler and hierarchical update logic
├── notebooks/         Exploratory analysis and visualizations
├── visuals/           Output plots (posterior distributions, shrinkage, etc.)
├── data/              Sample datasets (synthetic + real)
└── README.md          Project overview
```
---

## 🚧 Status

Mathematical derivation is complete. Code implementation is ongoing and will be uploaded progressively over the coming weeks.

---

## 📜 License

MIT
