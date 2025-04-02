# Bayesian Hierarchical Linear Regression (In Progress)

This project implements Bayesian linear regression using conjugate priors and a custom Gibbs sampler, then extends the model to a hierarchical prior structure to account for group-level variation. Everything is built from first principles in Python—no probabilistic programming libraries (e.g. PyMC3, Stan) are used.

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
  - Hierarchical priors:  
    \[
    \beta_j \sim \mathcal{N}(\mu_j, \tau^2), \quad \mu_j \sim \mathcal{N}(\mu_0, \sigma^2)
    \]
- Diagnostic tools:
  - Trace plots
  - Posterior predictive checks
  - Autocorrelation plots

---

## 🗂 Project Structure (coming soon)

