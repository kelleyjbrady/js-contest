import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd


df


def bayesian_saturation_model(df):
    """
    Fits an asymptotic exponential scaling law to GCG saturation data using PyMC.
    Model: S(N) = S_inf - alpha * exp(-beta * N)
    """
    # Extract data
    N_obs = df["sequence_length"].values
    S_obs = df["joint_score"].values

    print("[*] Compiling PyMC Model: Asymptotic Scaling Law...")
    with pm.Model() as asymptotic_model:
        # 1. Priors (Informative but weakly bound based on our domain knowledge)
        # S_inf: The theoretical ceiling. We know it's > 0 and likely < 0.5 for a 671B model.
        S_inf = pm.Normal("S_inf", mu=0.3, sigma=0.15)

        # alpha: The recoverable margin (must be positive)
        alpha = pm.HalfNormal("alpha", sigma=0.2)

        # beta: The decay rate (must be positive)
        beta = pm.HalfNormal("beta", sigma=0.1)

        # sigma: Observation noise
        sigma = pm.HalfNormal("sigma", sigma=0.05)

        # 2. Expected Value (The Deterministic Core)
        mu = pm.Deterministic("mu", S_inf - alpha * pm.math.exp(-beta * N_obs))

        # 3. Likelihood (Using Student-T instead of Normal to robustly handle GCG noise spikes)
        nu = pm.Exponential("nu", 1 / 15.0) + 1.0  # Degrees of freedom
        Y_obs = pm.StudentT("Y_obs", nu=nu, mu=mu, sigma=sigma, observed=S_obs)

        # 4. Inference (MCMC Sampling via NUTS)
        print("[*] Sampling Posterior Distribution...")
        trace = pm.sample(
            draws=2000,
            tune=1500,
            chains=4,
            target_accept=0.95,
            random_seed=42,
            return_inferencedata=True,
        )

        # 5. Posterior Predictive Checks (To draw the Credible Intervals on the graph)
        print("[*] Generating Posterior Predictive Samples for HDIs...")
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    # Output the Bayesian Summary
    print("\n==================================================")
    print("[*] BAYESIAN PARAMETER ESTIMATION (94% HDI)")
    print("==================================================")
    summary = az.summary(trace, var_names=["S_inf", "alpha", "beta", "sigma"])
    print(summary)

    return trace, asymptotic_model


# Example of how to plot the results with ArviZ:
# az.plot_trace(trace, var_names=['S_inf', 'alpha', 'beta'])
# plt.tight_layout()
# plt.show()
