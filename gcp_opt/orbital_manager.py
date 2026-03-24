import subprocess
import time
import re
import sys
import os

# --- CONFIGURATION ---
PROJECT_ID = "js-puzzle-491119"
BUCKET_NAME = f"{PROJECT_ID}-gcg-data"
IMAGE_NAME = f"gcr.io/{PROJECT_ID}/gcg-optimizer:latest"
INSTANCE_NAME = "gcg-spot-node-1"
TARGET_MAX_LEN = 60

# Strategy 1: The L4 Hit List
L4_ZONES = ["us-west1-a", "us-west1-b", "us-central1-a", "us-central1-c", "us-east1-d"]

# Strategy 2: The T4 Fallback List
T4_ZONES = ["us-west1-b", "us-central1-a", "us-east1-c"]


def run_command(cmd, shell=False, silent=False):
    """Executes a command and returns the return code, stdout, and stderr."""
    if not silent:
        display_cmd = cmd if isinstance(cmd, str) else " ".join(cmd)
        print(f"\n[EXEC] {display_cmd}")

    # If cmd is a string, we must use shell=True. If it's a list, shell=False is safer.
    use_shell = isinstance(cmd, str)
    result = subprocess.run(cmd, shell=use_shell, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def setup_environment():
    print("[*] Phase 1: Preparing Orbital Payload...")
    run_command(f"gcloud config set project {PROJECT_ID}")

    code, _, err = run_command(f"docker build -t {IMAGE_NAME} .")
    if code != 0:
        print(f"[!] Docker build failed: {err}")
        sys.exit(1)

    code, _, err = run_command(f"docker push {IMAGE_NAME}")
    if code != 0:
        print(f"[!] Docker push failed: {err}")
        sys.exit(1)

    startup_script = f"""#!/bin/bash
mkdir -p /var/mnt/disks/data /var/mnt/disks/output
gcloud storage cp gs://{BUCKET_NAME}/*.pt /var/mnt/disks/data/
chmod -R 777 /var/mnt/disks/data /var/mnt/disks/output
"""
    with open("/tmp/startup.sh", "w") as f:
        f.write(startup_script)
    print("[+] Payload staged successfully.")


def get_resume_length():
    code, out, _ = run_command(
        f"gcloud storage ls gs://{BUCKET_NAME}/logs/", silent=True
    )
    if code != 0 or not out:
        return 5

    lengths = [int(m) for m in re.findall(r"_L(\d+)_", out)]
    return max(lengths) if lengths else 5


def launch_instance(zone, machine_type, gpu_type, min_len):
    print(f"\n[*] Phase 2: Requesting {gpu_type} in {zone} (Resuming at L{min_len})...")

    # Using a list entirely prevents Bash trailing-space line-break errors
    cmd_list = [
        "gcloud",
        "compute",
        "instances",
        "create-with-container",
        INSTANCE_NAME,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        f"--accelerator=count=1,type={gpu_type}",
        "--image-family=cos-stable",
        "--image-project=cos-cloud",
        "--boot-disk-size=50GB",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        "--metadata-from-file=startup-script=/tmp/startup.sh",
        f"--container-image={IMAGE_NAME}",
        "--container-mount-host-path=host-path=/var/mnt/disks/data,mount-path=/app/data,mode=rw",
        "--container-mount-host-path=host-path=/var/mnt/disks/output,mount-path=/app/output,mode=rw",
        f"--container-arg=--project_id={PROJECT_ID}",
        f"--container-arg=--min_len={min_len}",
        f"--container-arg=--max_len={TARGET_MAX_LEN}",
    ]

    code, out, err = run_command(cmd_list)

    if code == 0:
        print(f"[+] SUCCESS! VM provisioned in {zone}.")
        return "SUCCESS"

    # --- STRICT ERROR PARSING ---
    if "Quota" in err and "exceeded" in err:
        print(
            f"\n[FATAL QUOTA LOCKOUT] Google is blocking the hardware request in {zone}."
        )
        print(f"Error Details: {err}")
        print("\nFix: Go to GCP Console -> 'IAM & Admin' -> 'Quotas'.")
        print(
            "Search for 'Preemptible NVIDIA L4 GPUs' or 'GPUs (all regions)' and request an increase from 0 to 1."
        )
        sys.exit(1)

    if "already exists" in err:
        print(
            f"[-] Dead instance {INSTANCE_NAME} found in zone. Terminating it to clear the runway..."
        )
        run_command(
            f"gcloud compute instances delete {INSTANCE_NAME} --zone={zone} --quiet"
        )
        return "RETRY_SAME"

    if "not enabled on project" in err:
        print("[-] Compute API not enabled. Enabling now (this takes ~60s)...")
        run_command("gcloud services enable compute.googleapis.com")
        return "RETRY_SAME"

    if (
        "ZONE_RESOURCE_POOL_EXHAUSTED" in err
        or "stockout" in err.lower()
        or "does not have enough resources" in err.lower()
    ):
        print(f"[-] Genuine stockout in {zone}.")
        return "STOCKOUT"

    print(f"\n[!] Unhandled error provisioning VM:\n{err}")
    return "UNKNOWN_ERROR"


def monitor_instance(zone):
    """Polls the VM status and actively streams Docker output."""
    print(f"\n[*] Phase 3: Establishing SSH Telemetry to {INSTANCE_NAME} in {zone}...")

    # Give the VM 20 seconds to finish booting the SSH daemon
    time.sleep(20)

    # This bash command waits for the container to exist, then tails its logs
    docker_wait_and_tail = f"""
    until sudo docker ps -qf ancestor={IMAGE_NAME} | grep -q .; do sleep 2; done; 
    sudo docker logs -f $(sudo docker ps -qf ancestor={IMAGE_NAME} | head -n 1)
    """

    ssh_cmd = [
        "gcloud",
        "compute",
        "ssh",
        INSTANCE_NAME,
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--command",
        docker_wait_and_tail,
    ]

    print(
        "[*] Tapping into live Docker output (Press Ctrl+C to detach without killing VM)..."
    )
    print("-" * 60)
    try:
        # We use subprocess.run without capturing output so it pipes directly to your screen
        subprocess.run(ssh_cmd)
    except KeyboardInterrupt:
        print("\n[*] Detached from live logs. VM continues in the background.")

    print("-" * 60)
    print("[*] Falling back to silent health polling...")

    while True:
        time.sleep(30)
        code, out, err = run_command(
            f"gcloud compute instances describe {INSTANCE_NAME} --zone={zone} --format='value(status)'",
            silent=True,
        )

        if code != 0 or "was not found" in err:
            print(
                f"\n[-] VM {INSTANCE_NAME} no longer exists. It either finished or was preempted."
            )
            return
        else:
            sys.stdout.write(".")
            sys.stdout.flush()


def main():
    setup_environment()

    while True:
        min_len = get_resume_length()
        if min_len >= TARGET_MAX_LEN:
            print(
                f"\n[+] Optimization fully complete! Final length L{TARGET_MAX_LEN} reached."
            )
            break

        deployed = False
        for zone in L4_ZONES:
            status = launch_instance(zone, "g2-standard-4", "nvidia-l4", min_len)
            if status == "RETRY_SAME":
                status = launch_instance(zone, "g2-standard-4", "nvidia-l4", min_len)

            if status == "SUCCESS":
                monitor_instance(zone)
                deployed = True
                break

        if deployed:
            continue

        print("\n[*] ALL L4 ZONES EXHAUSTED. Initiating T4 Downgrade Protocol...")
        for zone in T4_ZONES:
            status = launch_instance(zone, "n1-standard-4", "nvidia-tesla-t4", min_len)
            if status == "SUCCESS":
                monitor_instance(zone)
                deployed = True
                break

        if deployed:
            continue

        print("\n[!] CRITICAL: Complete global stockout of both L4 and T4 GPUs.")
        print("Sleeping for 10 minutes before retrying...")
        time.sleep(600)


if __name__ == "__main__":
    main()
