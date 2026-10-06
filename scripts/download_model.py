#!/usr/bin/env python3
"""
High-Speed Gemma 4 Model Fetcher for Cloud Environments.

Uses `huggingface_hub.snapshot_download` with high-throughput transfer
to pull Gemma 4 model weights directly to the target directory over
datacenter networking (~10 Gbps).
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Fast automated model downloader for Gemma 4."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="google/gemma-4-E2B-it",
        help="Hugging Face repository ID (default: google/gemma-4-E2B-it)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="models/gemma-4-e2b",
        help="Target local directory for weights (default: models/gemma-4-e2b)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face Access Token (defaults to HF_TOKEN env var)",
    )

    args = parser.parse_args()

    # Enable fast downloads if available
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    target_path = Path(args.local_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Gemma 4 Model Downloader ===")
    print(f"Repository : {args.repo_id}")
    print(f"Target Path: {target_path}")

    # Check if files already exist
    expected_files = ["config.json", "tokenizer.json", "model.safetensors"]
    existing_files = [f for f in expected_files if (target_path / f).exists()]

    if len(existing_files) == len(expected_files):
        print(f"✓ All expected model files already exist in {target_path}. Skipping download.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print("Fetching repository snapshot from Hugging Face...")
    try:
        downloaded_dir = snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(target_path),
            token=args.token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print(f"✓ Snapshot downloaded successfully to: {downloaded_dir}")
    except Exception as e:
        print(f"✗ Failed to download model snapshot: {e}", file=sys.stderr)
        sys.exit(1)

    # Post-download verification
    missing = [f for f in expected_files if not (target_path / f).exists()]
    if missing:
        print(f"Warning: Expected file(s) missing after download: {missing}", file=sys.stderr)
    else:
        print("✓ Verification passed: config.json, tokenizer.json, and model.safetensors present.")


if __name__ == "__main__":
    main()
