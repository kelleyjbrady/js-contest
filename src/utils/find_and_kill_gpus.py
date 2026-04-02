import google.auth
from googleapiclient import discovery
from googleapiclient.errors import HttpError


def hunt_for_gpus():
    # Authenticate using your local gcloud credentials
    try:
        credentials, _ = google.auth.default()
    except Exception as e:
        print(
            f"[!] Authentication failed. Did you run 'gcloud auth application-default login'?\n{e}"
        )
        return

    # Build the Resource Manager and Compute Engine services
    try:
        rm_service = discovery.build(
            "cloudresourcemanager", "v1", credentials=credentials
        )
        compute_service = discovery.build("compute", "v1", credentials=credentials)
    except Exception as e:
        print(f"[!] Failed to build GCP services: {e}")
        return

    print("[*] Fetching all accessible GCP Projects...")
    projects = []
    try:
        request = rm_service.projects().list()
        while request is not None:
            response = request.execute()
            for project in response.get("projects", []):
                # Only check active projects
                if project.get("lifecycleState") == "ACTIVE":
                    projects.append(project["projectId"])
            request = rm_service.projects().list_next(
                previous_request=request, previous_response=response
            )
    except HttpError as e:
        print(f"[!] Error fetching projects: {e}")
        return

    print(f"[*] Found {len(projects)} active projects. Hunting for GPUs...\n")

    found_gpus = False

    for project_id in projects:
        try:
            # aggregatedList searches all zones within the project simultaneously
            request = compute_service.instances().aggregatedList(project=project_id)

            while request is not None:
                response = request.execute()

                for zone, instances_scoped_list in response.get("items", {}).items():
                    instances = instances_scoped_list.get("instances", [])
                    for instance in instances:
                        name = instance.get("name")
                        status = instance.get("status")
                        machine_type = instance.get("machineType", "").split("/")[-1]

                        # Check for attached GPUs (Accelerators)
                        accelerators = instance.get("guestAccelerators", [])

                        if accelerators:
                            found_gpus = True
                            clean_zone = zone.split("/")[-1]
                            print(f"🚨 FOUND GPU INSTANCE 🚨")
                            print(f"  Project: {project_id}")
                            print(f"  Zone:    {clean_zone}")
                            print(f"  Name:    {name}")
                            print(f"  Status:  {status}")
                            print(f"  Machine: {machine_type}")
                            for acc in accelerators:
                                acc_type = acc.get("acceleratorType", "").split("/")[-1]
                                acc_count = acc.get("acceleratorCount", 0)
                                print(f"  => Attached: {acc_count}x {acc_type}")
                            print("-" * 40)

                request = compute_service.instances().aggregatedList_next(
                    previous_request=request, previous_response=response
                )

        except HttpError as e:
            # 403 usually means the Compute API isn't enabled on that specific project. We can safely skip.
            if e.resp.status == 403:
                continue
            else:
                print(f"[!] Error scanning project {project_id}: {e}")

    if not found_gpus:
        print(
            "\n[+] Scan complete. No instances with attached GPUs found in standard Compute Engine paths."
        )
        print(
            "[!] REMINDER: If the quota is still active, check Google Kubernetes Engine (GKE) or Cloud Run, as those manage nodes differently."
        )


if __name__ == "__main__":
    hunt_for_gpus()
