# Bayesian Hierarchical Linear Regression (In Progress)

This project implements **Bayesian linear regression** using **conjugate priors** and a custom **Gibbs sampler**, then extends the model to a **hierarchical prior structure** to account for group-level variation. The project is entirely developed from first principles in Python, without relying on external probabilistic programming libraries.

## 📚 Objectives

- Derive the full posterior distribution for a linear regression model using Normal-Inverse-Gamma conjugate priors
- Implement a Gibbs sampler to estimate posterior distributions for regression coefficients and noise variance
- Extend the model to incorporate **hierarchical priors** for multilevel group structure
- Visualize posterior distributions, predictive uncertainty, and shrinkage effects
- Compare Bayesian inference to frequentist OLS estimates and assess robustness to prior choices

## 🔧 Features

- Synthetic and real-world dataset support (e.g., Boston Housing)
- Custom implementation of:
  - Closed-form Bayesian updates
  - Gibbs sampling algorithm
  - Hierarchical priors:  
    \[
    \beta_j \sim \mathcal{N}(\mu_j, \tau^2), \quad \mu_j \sim \mathcal{N}(\mu_0, \sigma^2)
    \]
- Diagnostic tools: trace plots, autocorrelation, posterior predictive checks

## 🗂 Structure (coming soon)

- `inference/`: Gibbs sampler, hierarchical updates
- `notebooks/`: Exploratory analysis & visualizations
- `visuals/`: Plots of posteriors and model outputs
- `data/`: Sample datasets
- `README.md`: This file

## 🧠 Status

🟡 **Implementation in progress**  
Sampler functional for basic regression; hierarchical model integration underway.


