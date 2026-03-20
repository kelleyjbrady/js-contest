import json
import os
import pandas as pd
import numpy as np


INPUT_FILE = "/app/data/activations/combined_parquet/20260320_001643_batched/gcg_trigger_search_20260320_022746.jsonl"
OUTPUT_FILE = "/app/data/activations/combined_parquet/20260320_001643_batched/enriched_gcg_trigger_search_20260320_022746.jsonl"


PATIENCE_THRESHOLD = 50


def enrich_gcg_logs_pandas(input_path: str, output_path: str, patience_threshold: int):
    print(f"[*] Loading logs into Pandas from: {input_path}")

    # 1. Load the JSONL into a DataFrame
    df = pd.read_json(input_path, lines=True)

    # 2. Rename original step to absolute_step
    df = df.rename(columns={"step": "absolute_step"})

    # Ensure data is perfectly sorted chronologically within each batch
    df = df.sort_values(by=["sequence_length", "init_type", "absolute_step"])

    # 3. Detect the Annealing Shock
    # Calculate the jump between consecutive absolute_steps per batch
    df["step_jump"] = df.groupby(["sequence_length", "init_type"])[
        "absolute_step"
    ].diff()

    # Flag the exact row where the patience limit was triggered
    df["is_transition"] = df["step_jump"] > patience_threshold

    # Cumulative sum creates a boolean mask for everything AFTER the transition
    df["in_ascii_phase"] = (
        df.groupby(["sequence_length", "init_type"])["is_transition"]
        .cumsum()
        .astype(bool)
    )

    # 4. Assign Phase Labels
    df["phase"] = np.where(df["in_ascii_phase"], "ascii_constrained", "unconstrained")

    # 5. Calculate ASCII-specific metrics
    # Find the starting absolute_step of the ASCII phase and broadcast it to the group
    df["transition_step"] = df["absolute_step"].where(df["is_transition"])
    df["first_ascii_step"] = df.groupby(["sequence_length", "init_type"])[
        "transition_step"
    ].transform("max")

    # within_ascii_step: Distance from the transition step
    df["within_ascii_step"] = np.where(
        df["in_ascii_phase"], df["absolute_step"] - df["first_ascii_step"], np.nan
    )

    # ascii_log_index: 0, 1, 2... chronological counter within the ASCII phase
    df["ascii_log_index"] = np.where(
        df["in_ascii_phase"],
        df.groupby(["sequence_length", "init_type", "phase"]).cumcount(),
        np.nan,
    )

    # 6. Create the unified 'step' column for downstream scripts
    df["step"] = np.where(
        df["in_ascii_phase"], df["within_ascii_step"], df["absolute_step"]
    )

    # 7. Cleanup and Formatting
    # Drop the temporary calculation columns
    df = df.drop(
        columns=[
            "step_jump",
            "is_transition",
            "in_ascii_phase",
            "transition_step",
            "first_ascii_step",
        ]
    )

    # Enforce nullable integer types so NaN values don't convert the whole column to floats
    int_cols = ["absolute_step", "step", "within_ascii_step", "ascii_log_index"]
    for col in int_cols:
        df[col] = df[col].astype("Int64")

    # Reorder columns for readability (putting our new step metrics up front)
    cols = df.columns.tolist()
    front_cols = [
        "timestamp",
        "sequence_length",
        "init_type",
        "phase",
        "absolute_step",
        "step",
    ]
    remaining = [c for c in cols if c not in front_cols]
    df = df[front_cols + remaining]

    # 8. Export to JSONL
    df.to_json(output_path, orient="records", lines=True)

    print(f"[+] Enrichment complete!")
    print(f"    - Total records processed: {len(df)}")
    print(
        f"    - ASCII transitions detected: {df[df['phase'] == 'ascii_constrained']['sequence_length'].count()}"
    )
    print(f"[+] Output saved to: {output_path}")


if __name__ == "__main__":
    enrich_gcg_logs_pandas(INPUT_FILE, OUTPUT_FILE, PATIENCE_THRESHOLD)
