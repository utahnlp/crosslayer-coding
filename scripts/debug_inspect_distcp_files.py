#!/usr/bin/env python3
"""
Directly inspect the contents of .distcp files to determine if they contain different data.
This bypasses the distributed loading mechanism.
"""

import os
import sys
import json
import torch
from pathlib import Path
import pickle

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def inspect_distcp_file(filepath):
    """Inspect a .distcp file directly."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")
    
    # Try different methods to load the file
    try:
        # Method 1: Try torch.load with weights_only=False
        print("\nTrying torch.load with weights_only=False...")
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        print(f"Success! Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Number of keys: {len(data)}")
            # Show first few keys
            for i, (key, value) in enumerate(list(data.items())[:5]):
                if hasattr(value, 'shape'):
                    checksum = torch.sum(torch.abs(value)).item()
                    print(f"  {key}: shape={value.shape}, checksum={checksum:.2f}")
                else:
                    print(f"  {key}: type={type(value)}")
            
            # Check specific encoder weight
            enc_key = "encoder_module.encoders.0.weight"
            if enc_key in data:
                tensor = data[enc_key]
                checksum = torch.sum(torch.abs(tensor)).item()
                sample = tensor.flatten()[:5].tolist()
                print(f"\nSpecific check - {enc_key}:")
                print(f"  Shape: {tensor.shape}")
                print(f"  Checksum: {checksum:.6f}")
                print(f"  First 5 values: {sample}")
                return checksum
                
    except Exception as e:
        print(f"torch.load failed: {e}")
        
    # Method 2: Try loading as raw pickle
    try:
        print("\nTrying pickle.load...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Success with pickle! Type: {type(data)}")
    except Exception as e:
        print(f"pickle.load failed: {e}")
    
    # Method 3: Check file size and header
    print(f"\nFile info:")
    print(f"  Size: {os.path.getsize(filepath):,} bytes")
    
    # Read first few bytes to check format
    with open(filepath, 'rb') as f:
        header = f.read(100)
        print(f"  First 20 bytes (hex): {header[:20].hex()}")
        
    return None


def main():
    # Paths
    output_dir = Path("./debug_weight_check")
    checkpoint_dir = output_dir / "latest"
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Find all .distcp files
    distcp_files = sorted(checkpoint_dir.glob("*.distcp"))
    print(f"\nFound {len(distcp_files)} .distcp files:")
    for f in distcp_files:
        print(f"  {f.name} ({os.path.getsize(f):,} bytes)")
    
    # Inspect each file
    checksums = {}
    for distcp_file in distcp_files:
        checksum = inspect_distcp_file(distcp_file)
        if checksum is not None:
            checksums[distcp_file.name] = checksum
    
    # Compare checksums
    if len(checksums) == 2:
        print(f"\n{'='*60}")
        print("Checksum comparison:")
        print(f"{'='*60}")
        for name, checksum in checksums.items():
            print(f"{name}: {checksum:.6f}")
        
        values = list(checksums.values())
        if abs(values[0] - values[1]) < 0.01:
            print("\n⚠️  WARNING: Both .distcp files have the same encoder checksum!")
            print("This means the files contain identical data.")
        else:
            print("\n✅ Good: The .distcp files have different encoder checksums.")
            print("This means the files contain different data as expected.")
    
    # Also check the metadata file
    metadata_file = checkpoint_dir / ".metadata"
    if metadata_file.exists():
        print(f"\n{'='*60}")
        print("Checking .metadata file")
        print(f"{'='*60}")
        print(f"Size: {os.path.getsize(metadata_file):,} bytes")
        
        try:
            # The metadata file might be JSON or pickle
            with open(metadata_file, 'r') as f:
                content = f.read(200)
                print(f"First 200 chars: {content}")
        except:
            print("Could not read as text, might be binary format")


if __name__ == "__main__":
    main()