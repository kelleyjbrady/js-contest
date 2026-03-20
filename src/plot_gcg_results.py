import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Configuration
INPUT_FILE = "/app/data/activations/combined_parquet/20260320_001643_batched/enriched_gcg_trigger_search_20260320_022746.jsonl"
OUTPUT_HTML = "/app/data/activations/combined_parquet/20260320_001643_batched/gcg_manifold_visualization_20260320_022746.html"


def plot_3d_optimization_manifold(jsonl_path, output_path):
    print(f"[*] Loading enriched trajectories from {jsonl_path}...")
    df = pd.read_json(jsonl_path, lines=True)

    # Create a unified category for coloring the manifolds
    df["manifold_group"] = df["init_type"] + " | " + df["phase"]
    df = (
        df.groupby(["manifold_group", "step", "sequence_length"])["score"]
        .mean()
        .reset_index()
    )

    # Define a distinct color palette
    color_map = {
        "Warm Start | unconstrained": "rgba(255, 100, 100, 0.8)",  # Light Red
        "Warm Start | ascii_constrained": "rgba(180, 0, 0, 1.0)",  # Dark Red
        "Random Start | unconstrained": "rgba(100, 150, 255, 0.8)",  # Light Blue
        "Random Start | ascii_constrained": "rgba(0, 50, 200, 1.0)",  # Dark Blue
    }

    fig = go.Figure()

    # Group the data to plot individual lines (which visually form a wireframe manifold)
    grouped = df.groupby(["manifold_group", "sequence_length"])

    for (manifold_group, seq_length), group in grouped:
        # Sort chronologically to ensure the line draws correctly
        group = group.sort_values("step")

        # Add the 3D line trace
        fig.add_trace(
            go.Scatter3d(
                x=group["step"],
                y=group["sequence_length"],
                z=group["score"],
                mode="lines",
                line=dict(color=color_map.get(manifold_group, "gray"), width=4),
                name=f"{manifold_group} (L={seq_length})",
                # This handles the legend grouping so clicking one group toggles all its lines
                legendgroup=manifold_group,
                # showlegend=(
                #    seq_length == df["sequence_length"].min()
                # ),  # Only show one legend item per group
                # Interactive hover template
                # customdata=group[["decoded_string", "absolute_step"]],
                hovertemplate=(
                    "<b>Seq Length:</b> %{y}<br>"
                    + "<b>Relative Step:</b> %{x}<br>"
                    + "<b>Absolute Step:</b> %{customdata[1]}<br>"
                    + "<b>Score:</b> %{z:.4f}<br>"
                    + "<br><b>String:</b><br>%{customdata[0]}<br>"
                    + "<extra></extra>"  # Hides the secondary box
                ),
            )
        )

    # Format the 3D layout
    fig.update_layout(
        title="GCG Optimization Manifolds: Phase & Initialization Trajectories",
        scene=dict(
            xaxis_title="Relative Step (0 -> Max)",
            yaxis_title="Sequence Length (Depth)",
            zaxis_title="Cosine Similarity Score (Height)",
            # Optional: adjust camera angle for a better initial isometric view
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            title="Trajectory Type", yanchor="top", y=0.9, xanchor="left", x=0.05
        ),
        template="plotly_dark",  # Dark mode makes the neon lines pop
    )

    # Save to a self-contained HTML file
    fig.write_html(output_path)
    print(f"[+] 3D Interactive plot saved to {output_path}")
    print(f"    Open this file in any web browser to explore.")


if __name__ == "__main__":
    plot_3d_optimization_manifold(INPUT_FILE, OUTPUT_HTML)
