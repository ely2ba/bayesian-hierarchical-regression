import numpy as np

class BayesianHierarchicalLinearRegression:
    """
    Implements a hierarchical Bayesian linear regression with known group structure
    using a Normal-Inverse-Gamma prior for (beta, sigma^2) and a Normal-Inverse-Gamma
    prior for the group-level random intercept variances (tau^2).
    
    Model:
        y_i = X_i * beta + gamma_{group(i)} + epsilon_i,
        where epsilon_i ~ N(0, sigma^2)
        
    Hierarchy:
        beta | sigma^2 ~ Normal(mu0, sigma^2 * Lambda0^{-1})
        sigma^2 ~ InverseGamma(alpha0, beta0)
        
        gamma_j | tau^2 ~ Normal(0, tau^2)
        tau^2 ~ InverseGamma(alpha_tau, beta_tau)
        
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The design matrix of predictors.
    y : np.ndarray, shape (n_samples,)
        The response vector.
    groups : np.ndarray, shape (n_samples,)
        Group IDs (integers) for each observation.
    mu0 : np.ndarray, shape (n_features,)
        Prior mean for beta.
    Lambda0 : np.ndarray, shape (n_features, n_features)
        Prior precision matrix (inverse-covariance) for beta's Normal prior
        (must be positive definite).
    alpha0 : float
        Shape parameter of the InverseGamma prior for sigma^2.
    beta0 : float
        Rate (or scale) parameter of the InverseGamma prior for sigma^2.
    alpha_tau : float
        Shape parameter of the InverseGamma prior for tau^2.
    beta_tau : float
        Rate (or scale) parameter of the InverseGamma prior for tau^2.
    random_state : int, optional
        Random seed for reproducibility.
        
    Notes
    -----
    - This sampler updates beta, gamma, sigma^2, and tau^2 in separate steps.
    - Conjugacy allows each full conditional distribution to be a known closed form.
    - The user should call `run_gibbs` to perform iterations of the Gibbs sampler.
    - The resulting samples can be accessed via the `samples_` attribute after fitting.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        mu0: np.ndarray,
        Lambda0: np.ndarray,
        alpha0: float,
        beta0: float,
        alpha_tau: float,
        beta_tau: float,
        random_state: int = None
    ):
        # Set seed for reproducibility
        self.rng = np.random.default_rng(seed=random_state)
        
        # Data
        self.X = X  # shape (n, p)
        self.y = y  # shape (n,)
        self.groups = groups  # shape (n,)
        
        # Check dimensions
        self.n, self.p = X.shape
        if self.y.shape[0] != self.n:
            raise ValueError("Mismatch in number of samples between X and y.")
        if self.groups.shape[0] != self.n:
            raise ValueError("Mismatch in number of samples between X and group IDs.")
        
        # Identify number of groups and create a fast lookup
        self.unique_groups = np.unique(groups)
        self.J = len(self.unique_groups)  # number of groups
        self.group_idx = [np.where(groups == g)[0] for g in self.unique_groups]
        
        # Priors for (beta, sigma^2)
        self.mu0 = mu0          # shape (p,)
        self.Lambda0 = Lambda0  # shape (p, p)
        self.alpha0 = alpha0    # scalar
        self.beta0 = beta0      # scalar
        
        # Priors for (gamma_j, tau^2)
        self.alpha_tau = alpha_tau  # scalar
        self.beta_tau = beta_tau    # scalar
        
        # Internal placeholders for parameters
        self.beta_ = None    # shape (p,)
        self.gamma_ = None   # shape (J,)  random intercept for each group
        self.sigma2_ = None  # float
        self.tau2_ = None    # float
        
        # Storage for samples
        self.samples_ = {
            "beta": [],
            "gamma": [],
            "sigma2": [],
            "tau2": []
        }
        
    def _sample_beta(self):
        """
        Sample beta from its Normal full conditional:
        
        beta | (sigma^2, gamma, y) ~ Normal(mu_beta, Sigma_beta)
        
        where:
            Sigma_beta = (Lambda0 + (X^T X) / sigma^2)^{-1}
            mu_beta = Sigma_beta * [Lambda0 * mu0 + (X^T (y - gamma_g)) / sigma^2]
        """
        # Center y by subtracting group intercepts
        y_tilde = self.y - self.gamma_[self.groups]
        
        # Compute posterior precision
        precision_post = self.Lambda0 + (self.X.T @ self.X) / self.sigma2_
        
        # Invert to get posterior covariance
        cov_post = np.linalg.inv(precision_post)
        
        # Compute posterior mean
        mean_post = (
            cov_post @ (
                self.Lambda0 @ self.mu0
                + (self.X.T @ y_tilde) / self.sigma2_
            )
        )
        
        # Draw from the multivariate normal
        self.beta_ = self.rng.multivariate_normal(mean_post, cov_post)
    
    def _sample_gamma(self):
        """
        Sample each group intercept gamma_j from its Normal full conditional:
        
        gamma_j | (beta, sigma^2, tau^2, y_j) ~ Normal(mean_j, var_j)
        
        where:
            var_j = (n_j / sigma^2 + 1 / tau^2)^{-1}
            mean_j = var_j * [sum_{i in group j}(y_i - X_i * beta) / sigma^2]
        """
        for j, idx in enumerate(self.group_idx):
            # Residual for group j
            y_j = self.y[idx]
            X_j = self.X[idx, :]
            # Summation of (y_j - X_j * beta)
            residual_sum = np.sum(y_j - X_j @ self.beta_)
            # Posterior variance
            var_j = 1.0 / (len(idx) / self.sigma2_ + 1.0 / self.tau2_)
            # Posterior mean
            mean_j = var_j * (residual_sum / self.sigma2_)
            
            # Sample from Normal
            self.gamma_[j] = self.rng.normal(mean_j, np.sqrt(var_j))
    
    def _sample_sigma2(self):
        """
        Sample sigma^2 from its Inverse-Gamma full conditional:
        
        sigma^2 | (beta, gamma, y) ~ InverseGamma(alpha0 + n/2,
                                                 beta0 + 0.5 * sum((y - X*beta - gamma_g)^2))
        
        where alpha0, beta0 are the prior parameters.
        """
        # Compute residual sum of squares
        residuals = self.y - (self.X @ self.beta_ + self.gamma_[self.groups])
        rss = np.sum(residuals ** 2)
        
        # Posterior shape
        alpha_post = self.alpha0 + self.n / 2.0
        # Posterior rate (or scale, depending on parameterization)
        beta_post = self.beta0 + 0.5 * rss
        
        # Sample from InverseGamma by using Gamma sampling:
        # If X ~ Gamma(shape=alpha_post, scale=1/beta_post),
        # then 1/X ~ InverseGamma(alpha=alpha_post, beta=beta_post).
        gamma_sample = self.rng.gamma(shape=alpha_post, scale=1.0 / beta_post)
        self.sigma2_ = 1.0 / gamma_sample
    
    def _sample_tau2(self):
        """
        Sample tau^2 from its Inverse-Gamma full conditional:
        
        tau^2 | (gamma) ~ InverseGamma(alpha_tau + J/2,
                                      beta_tau + 0.5 * sum(gamma_j^2))
        """
        # Sum of squares of gamma_j
        sum_gamma_sq = np.sum(self.gamma_ ** 2)
        
        # Posterior shape
        alpha_post = self.alpha_tau + self.J / 2.0
        # Posterior rate
        beta_post = self.beta_tau + 0.5 * sum_gamma_sq
        
        gamma_sample = self.rng.gamma(shape=alpha_post, scale=1.0 / beta_post)
        self.tau2_ = 1.0 / gamma_sample

    def run_gibbs(self, n_iter=1000, burn_in=500):
        """
        Execute the Gibbs sampling routine for a specified number of iterations.
        
        Parameters
        ----------
        n_iter : int
            Total number of Gibbs iterations to perform.
        burn_in : int
            Number of initial samples to discard as burn-in.
        
        Returns
        -------
        self : BayesianHierarchicalLinearRegression
            The fitted sampler with stored samples in `samples_`.
        """
        # ===== Initialization =====
        # A simple strategy is to initialize with zeros or small random values.
        if self.beta_ is None:
            self.beta_ = self.rng.normal(size=self.p)
        if self.gamma_ is None:
            self.gamma_ = self.rng.normal(size=self.J)
        if self.sigma2_ is None:
            self.sigma2_ = 1.0
        if self.tau2_ is None:
            self.tau2_ = 1.0
        
        # ===== Main Gibbs Loop =====
        for it in range(n_iter):
            # 1. Sample beta
            self._sample_beta()
            # 2. Sample gamma
            self._sample_gamma()
            # 3. Sample sigma^2
            self._sample_sigma2()
            # 4. Sample tau^2
            self._sample_tau2()
            
            # Store samples if past burn-in
            if it >= burn_in:
                self.samples_["beta"].append(self.beta_.copy())
                self.samples_["gamma"].append(self.gamma_.copy())
                self.samples_["sigma2"].append(self.sigma2_)
                self.samples_["tau2"].append(self.tau2_)
        
        # Convert lists to numpy arrays for convenience
        self.samples_["beta"] = np.array(self.samples_["beta"])
        self.samples_["gamma"] = np.array(self.samples_["gamma"])
        self.samples_["sigma2"] = np.array(self.samples_["sigma2"])
        self.samples_["tau2"] = np.array(self.samples_["tau2"])
        
        return self


if __name__ == "__main__":
    # =============== Example Usage (Synthetic Data) ===============
    # Suppose we have n=100 observations, p=2 features, and J=5 groups.
    rng = np.random.default_rng(seed=42)
    n, p, J = 100, 2, 5

    # Generate some synthetic data
    X_synth = rng.normal(size=(n, p))
    true_beta = np.array([1.0, -2.0])
    group_labels = rng.integers(low=0, high=J, size=n)

    # True group intercepts
    true_gamma = rng.normal(0, 1, size=J)

    # Construct y
    y_synth = (X_synth @ true_beta) + true_gamma[group_labels] + rng.normal(0, 1, size=n)

    # Prior hyperparameters (somewhat arbitrary)
    mu0 = np.zeros(p)
    Lambda0 = np.eye(p) * 0.01
    alpha0, beta0 = 2.0, 2.0
    alpha_tau, beta_tau = 2.0, 2.0

    # Create Gibbs sampler instance
    sampler = BayesianHierarchicalLinearRegression(
        X=X_synth,
        y=y_synth,
        groups=group_labels,
        mu0=mu0,
        Lambda0=Lambda0,
        alpha0=alpha0,
        beta0=beta0,
        alpha_tau=alpha_tau,
        beta_tau=beta_tau,
        random_state=42
    )

    # Run Gibbs sampler
    sampler.run_gibbs(n_iter=2000, burn_in=1000)

    # Access posterior samples
    beta_samples = sampler.samples_["beta"]     # shape (2000 - 1000, p)
    gamma_samples = sampler.samples_["gamma"]   # shape (2000 - 1000, J)
    sigma2_samples = sampler.samples_["sigma2"] # shape (2000 - 1000,)
    tau2_samples = sampler.samples_["tau2"]     # shape (2000 - 1000,)

    # Print a quick summary of posterior means
    print("Posterior mean of beta:", beta_samples.mean(axis=0))
    print("Posterior mean of sigma^2:", sigma2_samples.mean())
    print("Posterior mean of tau^2:", tau2_samples.mean())
    print("Posterior mean of gamma:", gamma_samples.mean(axis=0))
