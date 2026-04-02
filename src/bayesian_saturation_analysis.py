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
CAMPAIGN = "layer15_20_35_55_trigger_exec_raw_isoforest_deep_sweep_01"
CAMPAIGN_PRETTY_STR = "Trigger Exec Raw Layers 15, 20, 35, 55"
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
    df_raw = df.loc[df["sequence_length"] < max_len, :].reset_index(drop=True).copy()
    df = df_raw.groupby("sequence_length")["joint_score"].max().reset_index().copy()
    return df, df_raw


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


import numpy as np
import matplotlib.pyplot as plt
import arviz as az


def plot_bayesian_saturation(
    df_raw, df_frontier, trace, n_min=None, n_max=None, title_suffix="", save_path=None
):
    """
    Plots the GCG Pareto frontier against the Bayesian posterior predictive curve,
    while displaying the raw search variance as a background cloud.
    """
    print(f"\n[*] Generating Bayesian Posterior Predictive Plot for {title_suffix}...")

    # 1. Extract frontier data (for the red dots)
    x_frontier = df_frontier["sequence_length"].values
    y_frontier = df_frontier["joint_score"].values

    # Extract raw data (for the background cloud)
    x_raw = df_raw["sequence_length"].values
    y_raw = df_raw["joint_score"].values

    # 2. Extract flattened posterior samples
    S_inf_samples = trace.posterior["S_inf"].values.flatten()
    alpha_samples = trace.posterior["alpha"].values.flatten()
    beta_samples = trace.posterior["beta"].values.flatten()

    # 3. Continuous grid
    x_grid = np.linspace(x_frontier.min(), x_frontier.max() + 15, 500)

    # 4. Vectorized calculation of posterior curves (3-Parameter Model)
    posterior_curves = S_inf_samples[:, None] - alpha_samples[:, None] * np.exp(
        -beta_samples[:, None] * x_grid[None, :]
    )

    # 5. Extract the median and 94% HDI bounds
    curve_lower, curve_median, curve_upper = np.percentile(
        posterior_curves, [3, 50, 97], axis=0
    )

    # Calculate Bayesian R-squared
    y_pred_flat = trace.posterior_predictive["obs"].values.reshape(-1, len(y_frontier))

    # Calculate Bayesian R-squared across all flattened samples
    r2_dist = az.r2_score(y_true=y_frontier, y_pred=y_pred_flat)
    r2_median = r2_dist["r2"]

    # ==========================================
    # Matplotlib Configuration
    # ==========================================
    plt.figure(figsize=(12, 7))
    plt.style.use("seaborn-v0_8-darkgrid")

    # Plot the raw search variance (The "Cloud")
    plt.scatter(
        x_raw,
        y_raw,
        color="slategray",
        alpha=0.15,
        s=10,
        zorder=1,
        label="Sub-optimal GCG Steps (Search Variance)",
    )

    # Plot the 94% HDI shadow
    plt.fill_between(
        x_grid,
        curve_lower,
        curve_upper,
        color="royalblue",
        alpha=0.3,
        zorder=2,
        label="94% HDI (Posterior Uncertainty)",
    )

    # Plot the median predictive curve
    plt.plot(
        x_grid,
        curve_median,
        color="midnightblue",
        linewidth=2.5,
        zorder=3,
        label="Bayesian Median Asymptote",
    )

    # Scatter the Pareto frontier (The Red Dots)
    plt.scatter(
        x_frontier,
        y_frontier,
        color="crimson",
        edgecolor="white",
        s=45,
        zorder=5,
        label="Pareto Frontier (Max Achieved)",
    )

    # Plot the calculated optimization bounds
    if n_min is not None and n_max is not None:
        plt.axvspan(
            n_min,
            n_max,
            color="goldenrod",
            alpha=0.15,
            zorder=0,
            label="Optimal Payload Window",
        )
        plt.axvline(
            n_min,
            color="darkorange",
            linestyle="--",
            linewidth=2,
            zorder=4,
            label=f"$N_{{min}}$ (Elbow: {n_min})",
        )
        plt.axvline(
            n_max,
            color="darkred",
            linestyle="--",
            linewidth=2,
            zorder=4,
            label=f"$N_{{max}}$ (Saturation: {n_max})",
        )

    # Formatting and labels
    # Dynamically inject the campaign/layer name into the title
    plt.title(
        f"GCG Asymptotic Capacity Limit: {title_suffix}",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Sequence Length (Tokens)", fontsize=12, fontweight="bold")
    plt.ylabel("Target Cosine Similarity", fontsize=12, fontweight="bold")
    plt.xlim(max(0, x_frontier.min() - 5), x_frontier.max() + 10)

    # Plot the theoretical absolute ceiling & R2
    median_S_inf = np.median(S_inf_samples)
    plt.axhline(
        median_S_inf,
        color="black",
        linestyle=":",
        alpha=0.6,
        label=f"Theoretical Ceiling ($S_{{\infty}}$ ≈ {median_S_inf:.3f})\nBayesian $R^2$ ≈ {r2_median:.3f}",
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


df, df_raw = load_df(TARGET_DIR)
if os.path.exists(TRACE_DUMP) and not OVERWRITE_JB:
    trace = jb.load(TRACE_DUMP)
else:
    trace, mod = bayesian_saturation_model(df, log_file=LOG_FILE)
    with open(TRACE_DUMP, "wb") as f:
        jb.dump(trace, f)
# Example of how to plot the results with ArviZ:
az.plot_trace(trace, var_names=["S_inf", "alpha", "beta"])
plt.tight_layout()
plt.savefig(TRACEPLOT_PATH, dpi=300)

n_min, n_max = calculate_bayesian_bounds(trace=trace)

hard_code_n_min = 18
hard_code_n_max = 35
plot_bayesian_saturation(
    df_raw,
    df,
    trace,
    n_min=n_min if hard_code_n_min is None else hard_code_n_min,
    n_max=n_max if hard_code_n_max is None else hard_code_n_max,
    title_suffix=CAMPAIGN_PRETTY_STR,
    save_path=PLOT_PATH,
)

isolate_core_payload(
    TARGET_DIR,
    min_tokens=n_min if hard_code_n_min is None else hard_code_n_min,
    max_tokens=n_max if hard_code_n_max is None else hard_code_n_max,
    campaign_name=CAMPAIGN,
    campaign_pretty_str=CAMPAIGN_PRETTY_STR,
    text_output_path=LOG_FILE,
    output_dir="app",
)
