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
ACTIVE_ZONE = None

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
    """Builds the container, pushes to GCR, and writes the native GPU startup script."""
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

    # THE UPGRADED STARTUP SCRIPT
    startup_script = f"""#!/bin/bash
# 1. Fetch the dynamic restart length from the VM's custom metadata
MIN_LEN=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/min_len")

# 2. Wait for the Deep Learning VM to ensure NVIDIA drivers are loaded
echo "[*] Waiting for NVIDIA hardware to initialize..."
until nvidia-smi; do sleep 5; done

# 3. Stage the Data
mkdir -p /var/mnt/disks/data /var/mnt/disks/output
gcloud storage cp gs://{BUCKET_NAME}/*.pt /var/mnt/disks/data/
chmod -R 777 /var/mnt/disks/data /var/mnt/disks/output

# 4. Authenticate Docker with Google Container Registry
gcloud auth configure-docker --quiet

# 5. Launch the Optimizer with EXPLICIT native GPU passthrough
sudo docker run -d --gpus all \\
  -v /var/mnt/disks/data:/app/data \\
  -v /var/mnt/disks/output:/app/output \\
  {IMAGE_NAME} \\
  --project_id={PROJECT_ID} \\
  --min_len=$MIN_LEN \\
  --max_len={TARGET_MAX_LEN}
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
    """Attempts to provision a Deep Learning VM in the specified zone."""
    print(f"\n[*] Phase 2: Requesting {gpu_type} in {zone} (Resuming at L{min_len})...")

    # We switch to standard 'create', use the Deep Learning image, and bump the disk to 100GB
    cmd_list = [
        "gcloud",
        "compute",
        "instances",
        "create",
        INSTANCE_NAME,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        f"--accelerator=count=1,type={gpu_type}",
        "--image-family=common-cu128-ubuntu-2204-nvidia-570",
        "--image-project=deeplearning-platform-release",
        "--boot-disk-size=200GB",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        "--metadata-from-file=startup-script=/tmp/startup.sh",
        f"--metadata=install-nvidia-driver=True,min_len={min_len}",
    ]

    code, out, err = run_command(cmd_list)

    if code == 0:
        print(f"[+] SUCCESS! Deep Learning VM provisioned in {zone}.")
        return "SUCCESS"

    # --- STRICT ERROR PARSING ---
    if "Quota" in err and "exceeded" in err:
        print(
            f"\n[FATAL QUOTA LOCKOUT] Google is blocking the hardware request in {zone}."
        )
        print(f"Error Details: {err}")
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
        print("[-] Compute API not enabled. Enabling now...")
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
    global ACTIVE_ZONE
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
            ACTIVE_ZONE = zone  # Track the zone we are trying
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
            ACTIVE_ZONE = zone  # Track the fallback zone
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
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] ORCHESTRATOR ABORTED BY USER (Ctrl+C).")
    finally:
        # The Dead Man's Switch: Always run this cleanup when the script exits
        if ACTIVE_ZONE:
            print(f"[*] Sweeping {ACTIVE_ZONE} for orphaned VMs...")
            subprocess.run(
                f"gcloud compute instances delete {INSTANCE_NAME} --zone={ACTIVE_ZONE} --quiet",
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            print("[+] Cleanup complete. Runway cleared.")
