# Nebius Cloud H100 Reproduction and Deployment Guide

---

## Live Active Deployment

* **Host:** `195.242.13.222`
* **GPU:** NVIDIA H100 80GB SXM5
* **Go API Gateway (SSE):** `http://195.242.13.222:8080`
* **Rust gRPC Worker:** `127.0.0.1:50051`

### 1-Line Live Test from Anywhere:
```bash
curl -N -X POST http://195.242.13.222:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain why Rust and Go make a great pair.", "max_tokens": 50}'
```

---

## 1. Fast Launch Command (Creating a New Instance)

```bash
nebius compute instance create '{
  "metadata": {
    "name": "gemma4-gpu-runner",
    "parent_id": "project-e00erzgkpr00c62qktp20v"
  },
  "spec": {
    "resources": {
      "platform": "gpu-h100-sxm",
      "preset": "1gpu-16vcpu-200gb"
    },
    "boot_disk": {
      "attach_mode": "READ_WRITE",
      "existing_disk": {
        "id": "computedisk-e00zd4fbv65kwcgr3t"
      }
    },
    "network_interfaces": [
      {
        "subnet_id": "vpcsubnet-e00ecnhq710dhq1bng",
        "name": "eth0",
        "ip_address": {},
        "public_ip_address": {}
      }
    ],
    "cloud_init_user_data": "#cloud-config\nusers:\n  - name: ubuntu\n    groups: sudo\n    sudo: ALL=(ALL) NOPASSWD:ALL\n    shell: /bin/bash\n    ssh_authorized_keys:\n      - ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAr72ljb+WpE6a2In45F8kaARb0rAVxBaBytSZBih1xn me@jorgeajimenez.com"
  }
}'
```

---

## 2. Syncing and Running the Production Multi-Tier Stack

```bash
# 1. Sync the codebase from your local machine
rsync -avz --exclude 'target' /Users/jorgeajimenez/repos/candle/ ubuntu@<VM_PUBLIC_IP>:~/candle/

# 2. SSH into the VM
ssh ubuntu@<VM_PUBLIC_IP>

# 3. Start the Rust gRPC Worker with native CUDA acceleration (Terminal 1)
cd ~/candle
cargo run --release --no-default-features --features cuda

# 4. Start the Go API Gateway (Terminal 2)
cd ~/candle/go-gateway
go run main.go

# 5. Stream Chat Completions over Server-Sent Events (Terminal 3)
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain why Rust and Go make a great pair.", "max_tokens": 64}'
```

---

## 3. Teardown / Stop Commands

To delete the VM when not in use:

```bash
nebius compute instance delete --id <INSTANCE_ID>
```
