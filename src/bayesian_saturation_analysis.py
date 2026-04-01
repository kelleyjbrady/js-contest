import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import joblib as jb
from analyze_gcg_results import isolate_core_payload

OVERWRITE_JB = True
CAMPAIGN = "layer15_20_35_55_trigger_exec_isoforest_deep_sweep_02"
TARGET_DIR = f"/app/telemetry_data/{CAMPAIGN}/"
LOG_FILE = f"bayesian_saturation_model_params_{CAMPAIGN}.txt"
TRACE_DUMP = f"bayesian_saturation_model_trace_{CAMPAIGN}.jb"
TRACEPLOT_PATH = f"bayesian_saturation_model_traceplot_{CAMPAIGN}.png"
PLOT_PATH = f"bayesian_saturation_model_{CAMPAIGN}.png"


def load_df(target_dir):
    files = glob.glob(os.path.join(target_dir, "*.jsonl"))
    df = [pd.read_json(f, lines=True) for f in files]
    df = pd.concat(df)
    max_len = df["sequence_length"].max()
    df = df.loc[df["sequence_length"] < max_len, :].reset_index(drop=True)
    df = df.groupby("sequence_length")["joint_score"].max().reset_index()
    return df


def bayesian_saturation_model(df, log_file="bayesian_bounds.log"):
    """
    Fits an asymptotic exponential scaling law to GCG saturation data using PyMC.
    Model: S(N) = S_inf - alpha * exp(-beta * N)
    """
    # Extract data
    N_obs = df["sequence_length"].values
    S_obs = df["joint_score"].values

    # Data-driven prior anchors
    max_score = np.max(S_obs)
    score_range = max_score - np.min(S_obs)

    print("[*] Compiling PyMC Model: 3-Parameter Asymptotic Scaling Law...")
    with pm.Model() as model:
        # Data-informed Priors (Prevents the sampler from wandering into deep space)
        S_inf = pm.Normal("S_inf", mu=max_score * 1.1, sigma=score_range)
        alpha = pm.HalfNormal("alpha", sigma=score_range * 2)
        beta = pm.HalfNormal("beta", sigma=0.1)

        # The Standard Exponential Equation
        mu = pm.Deterministic("mu", S_inf - alpha * pm.math.exp(-beta * N_obs))

        # Likelihood
        sigma = pm.HalfNormal("sigma", sigma=0.02)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=S_obs)

        # Sample
        trace = pm.sample(2000, tune=1500, target_accept=0.95, cores=4)

        # Posterior Predictive Checks
        print("[*] Generating Posterior Predictive Samples for HDIs...")
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    # Output the Bayesian Summary
    print("\n==================================================")
    print("[*] BAYESIAN PARAMETER ESTIMATION (94% HDI)")
    print("==================================================")
    summary = az.summary(trace, var_names=["S_inf", "alpha", "beta", "sigma"])
    print(summary)

    with open(log_file, "w") as f:
        f.write(summary.to_string())

    return trace, model


import numpy as np
import matplotlib.pyplot as plt


def plot_bayesian_saturation(df, trace, n_min=None, n_max=None, save_path=None):
    """
    Plots the raw GCG trajectory against the Bayesian posterior predictive curve.
    Includes the 94% HDI and overlays the derived optimization bounds.
    """
    print("\n[*] Generating Bayesian Posterior Predictive Plot...")

    # 1. Extract raw empirical data
    x_obs = df["sequence_length"].values
    y_obs = df["joint_score"].values

    # 2. Extract flattened posterior samples
    S_inf_samples = trace.posterior["S_inf"].values.flatten()
    alpha_samples = trace.posterior["alpha"].values.flatten()
    beta_samples = trace.posterior["beta"].values.flatten()

    # 3. Create a dense continuous grid for smooth curve plotting
    # Extend the grid slightly past the max observed length to show the asymptote
    x_grid = np.linspace(x_obs.min(), x_obs.max() + 15, 500)

    # 4. Vectorized calculation of all posterior curves
    # Broadcasting shapes: (samples, 1) and (1, grid_points) -> (samples, grid_points)
    posterior_curves = S_inf_samples[:, None] - alpha_samples[:, None] * np.exp(
        -beta_samples[:, None] * x_grid[None, :]
    )

    # 5. Extract the median and 94% HDI bounds (3rd and 97th percentiles)
    curve_lower, curve_median, curve_upper = np.percentile(
        posterior_curves, [3, 50, 97], axis=0
    )

    # ==========================================
    # Matplotlib Configuration
    # ==========================================
    plt.figure(figsize=(12, 7))
    plt.style.use("seaborn-v0_8-darkgrid")

    # Plot the 94% HDI shadow
    plt.fill_between(
        x_grid,
        curve_lower,
        curve_upper,
        color="royalblue",
        alpha=0.3,
        label="94% HDI (Posterior Uncertainty)",
    )

    # Plot the median predictive curve
    plt.plot(
        x_grid,
        curve_median,
        color="midnightblue",
        linewidth=2.5,
        label="Bayesian Median Asymptote",
    )

    # Scatter the raw GCG data points
    plt.scatter(
        x_obs,
        y_obs,
        color="crimson",
        edgecolor="white",
        s=40,
        zorder=5,
        label="Raw Optimization Trajectory",
    )

    # Plot the calculated optimization bounds (if provided)
    if n_min is not None and n_max is not None:
        # Highlight the payload extraction window
        plt.axvspan(
            n_min, n_max, color="goldenrod", alpha=0.15, label="Optimal Payload Window"
        )

        plt.axvline(
            n_min,
            color="darkorange",
            linestyle="--",
            linewidth=2,
            label=f"$N_{{min}}$ (Elbow: {n_min})",
        )
        plt.axvline(
            n_max,
            color="darkred",
            linestyle="--",
            linewidth=2,
            label=f"$N_{{max}}$ (Saturation: {n_max})",
        )

    # Formatting and labels
    plt.title(
        "Bayesian Modeling of GCG Asymptotic Optimization Limits",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Sequence Length (Tokens)", fontsize=12, fontweight="bold")
    plt.ylabel("Target Cosine Similarity", fontsize=12, fontweight="bold")
    plt.xlim(max(0, x_obs.min() - 5), x_obs.max() + 10)

    # Optional: Plot the theoretical absolute ceiling (median S_inf)
    median_S_inf = np.median(S_inf_samples)
    plt.axhline(
        median_S_inf,
        color="black",
        linestyle=":",
        alpha=0.6,
        label=f"Theoretical Ceiling ($S_{{\infty}}$ ≈ {median_S_inf:.3f})",
    )

    plt.legend(loc="lower right", frameon=True, fontsize=10, shadow=True)
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[*] Plot saved successfully to: {save_path}")
    else:
        plt.show()

    plt.close()


def calculate_bayesian_bounds(trace, epsilon=0.001, log_file="bayesian_bounds.log"):
    """
    Calculates the posterior distributions for the optimization bounds
    using the MCMC trace from the PyMC asymptotic model.
    """
    print("\n[*] CALCULATING BAYESIAN OPTIMIZATION BOUNDS...")

    # 1. Extract the flattened posterior samples
    alpha_samples = trace.posterior["alpha"].values.flatten()
    beta_samples = trace.posterior["beta"].values.flatten()

    # ==========================================
    # UPPER BOUND (N_max): Derivative Threshold
    # ==========================================
    # Filter out invalid samples to prevent log(>1) math errors
    valid_mask = (alpha_samples * beta_samples) > epsilon
    a_val = alpha_samples[valid_mask]
    b_val = beta_samples[valid_mask]

    N_max_samples = -(1.0 / b_val) * np.log(epsilon / (a_val * b_val))

    # Get median and 94% HDI
    N_max_median = np.median(N_max_samples)
    N_max_hdi = az.hdi(N_max_samples, hdi_prob=0.94)

    # ==========================================
    # LOWER BOUND (N_min): Continuous Secant Tangency (Kneedle Equivalent)
    # ==========================================
    N_start = 5.0

    # Calculate the Y values at N_start and N_max for each specific sampled curve
    # Note: S_inf cancels out when calculating the difference (S_f - S_0)
    # S_0 = -a * exp(-b * N_start)
    # S_f = -a * exp(-b * N_max)

    S_0 = -a_val * np.exp(-b_val * N_start)
    S_f = -a_val * np.exp(-b_val * N_max_samples)

    # Calculate the slope of the secant line connecting start and max saturation
    m_secant = (S_f - S_0) / (N_max_samples - N_start)

    # The elbow occurs where the first derivative equals the secant slope
    # a * b * exp(-b * N) = m_secant
    N_min_samples = -(1.0 / b_val) * np.log(m_secant / (a_val * b_val))

    # Get median and 94% HDI
    N_min_median = np.median(N_min_samples)
    N_min_hdi = az.hdi(N_min_samples, hdi_prob=0.94)

    # ==========================================
    # Output the Results
    # ==========================================
    print(f"\n--- LOWER BOUND (N_min): Continuous Geometric Elbow ---")
    print(f"  Median N_min: {N_min_median:.1f} tokens")
    print(f"  94% HDI:      [{N_min_hdi[0]:.1f}, {N_min_hdi[1]:.1f}]")

    print(f"\n--- UPPER BOUND (N_max): Saturation (ε = {epsilon}) ---")
    print(f"  Median N_max: {N_max_median:.1f} tokens")
    print(f"  94% HDI:      [{N_max_hdi[0]:.1f}, {N_max_hdi[1]:.1f}]")

    # Recommended deterministic bounds for your payload script
    safe_N_min = int(np.floor(N_min_median))
    safe_N_max = int(np.ceil(N_max_median))

    print(
        f"\n[+] Recommended Payload Analysis Window: N ∈ [{safe_N_min}, {safe_N_max}]"
    )

    with open(log_file, "a") as f:
        f.write(
            f"\n[+] Recommended Payload Analysis Window: N ∈ [{safe_N_min}, {safe_N_max}]\n"
        )

    return safe_N_min, safe_N_max


df = load_df(TARGET_DIR)
if os.path.exists(TRACE_DUMP) and not OVERWRITE_JB:
    trace = jb.load(TRACE_DUMP)
else:
    trace, mod = bayesian_saturation_model(df)
    with open(TRACE_DUMP, "wb") as f:
        jb.dump(trace, f)
# Example of how to plot the results with ArviZ:
az.plot_trace(trace, var_names=["S_inf", "alpha", "beta"])
plt.tight_layout()
plt.savefig(TRACEPLOT_PATH, dpi=300)

n_min, n_max = calculate_bayesian_bounds(trace=trace)

plot_bayesian_saturation(df, trace, n_min=n_min, n_max=n_max, save_path=PLOT_PATH)

isolate_core_payload(
    TARGET_DIR,
    min_tokens=n_min,
    max_tokens=n_max,
    campaign_name=CAMPAIGN,
    text_output_path=LOG_FILE,
    output_dir="app",
)
