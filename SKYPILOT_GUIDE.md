# SkyPilot Multi-Cloud Deployment Guide (Nebius and GCP)

SkyPilot is the orchestration standard for provisioning, running, and tearing down Gemma 4 infrastructure across **Nebius Cloud** and **Google Cloud Platform (GCP)**.

---

## 1. Setup and Verification

SkyPilot is installed via `uv tool install "skypilot[gcp,nebius]"`.

### Verify Enabled Providers:
```bash
sky check
```
* **Nebius:** Enabled (`Nebius: enabled [compute]`)
* **GCP:** Requires `gcloud auth login` and `gcloud auth application-default login`

---

## 2. Nebius Cloud Deployment

### Provision and Launch:
```bash
sky launch -c gemma4-nebius sky-nebius.yaml --yes
```

### Check Cluster Status:
```bash
sky status
```

### SSH into Cluster:
```bash
ssh gemma4-nebius
# or:
sky ssh gemma4-nebius
```

### Tail Logs:
```bash
sky logs gemma4-nebius
```

### Teardown (Stop Billing):
```bash
sky down gemma4-nebius --yes
```

---

## 3. Google Cloud Platform (GCP) Deployment

### GCP Authentication Setup:
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <YOUR_GCP_PROJECT_ID>
sky check gcp
```

### Provision and Launch:
```bash
sky launch -c gemma4-gcp sky-gcp.yaml --yes
```

### Check Cluster Status:
```bash
sky status
```

### Teardown (Stop Billing):
```bash
sky down gemma4-gcp --yes
```

---

## 4. SkyPilot Quick Reference Cheat Sheet

| Action | Command |
| :--- | :--- |
| **Launch Nebius** | `sky launch -c gemma4-nebius sky-nebius.yaml --yes` |
| **Launch GCP** | `sky launch -c gemma4-gcp sky-gcp.yaml --yes` |
| **List Running Clusters** | `sky status` |
| **Stop Cluster (Save Disk)** | `sky stop <cluster_name> --yes` |
| **Delete Cluster (Stop Billing)** | `sky down <cluster_name> --yes` |
| **Delete All Active Clusters** | `sky down --all --yes` |
| **Stream Inference from Client** | `./stream.sh "Your prompt here"` |
