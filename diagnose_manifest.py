import numpy as np
import argparse
import os

# Define the manifest dtype, matching the one in ActivationGenerator and ManifestActivationStore
MANIFEST_DTYPE = np.dtype([("chunk_id", np.int32), ("num_tokens", np.int32), ("offset", np.int64)])


def diagnose_manifest_file(manifest_path):
    """
    Reads an index.bin manifest file and prints statistics about its contents.
    """
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        return

    try:
        manifest_data = np.fromfile(manifest_path, dtype=MANIFEST_DTYPE)
    except Exception as e:
        print(f"Error reading manifest file {manifest_path}: {e}")
        return

    if manifest_data.size == 0:
        print(f"Manifest file {manifest_path} is empty or not in the expected format.")
        return

    num_entries = manifest_data.shape[0]
    chunk_ids = manifest_data["chunk_id"]
    num_tokens_values = manifest_data["num_tokens"]
    offsets = manifest_data["offset"]

    print(f"--- Manifest File Diagnostics for: {manifest_path} ---")
    print(f"Total entries: {num_entries}")

    if num_entries > 0:
        print("\nChunk ID Statistics:")
        print(f"  Min chunk_id: {np.min(chunk_ids)}")
        print(f"  Max chunk_id: {np.max(chunk_ids)}")
        if not np.all(np.diff(chunk_ids) == 1) and num_entries > 1:
            print("  Warning: Chunk IDs are not strictly sequential or contain duplicates!")
        else:
            print("  Chunk IDs appear sequential and unique.")

        print("\nNum Tokens Statistics:")
        print(f"  Min num_tokens: {np.min(num_tokens_values)}")
        print(f"  Max num_tokens: {np.max(num_tokens_values)}")
        print(f"  Mean num_tokens: {np.mean(num_tokens_values):.2f}")
        print(f"  Median num_tokens: {np.median(num_tokens_values)}")
        if np.median(num_tokens_values) < 100:  # Arbitrary low threshold to flag likely issues
            print(f"  WARNING: Median num_tokens ({np.median(num_tokens_values)}) is very low!")

        print("\nOffset Statistics:")
        print(f"  Min offset: {np.min(offsets)}")
        print(f"  Max offset: {np.max(offsets)}")

        print("\nFirst 5 entries:")
        for i in range(min(5, num_entries)):
            print(
                f"  Entry {i}: chunk_id={manifest_data[i]['chunk_id']}, num_tokens={manifest_data[i]['num_tokens']}, offset={manifest_data[i]['offset']}"
            )

        print("\nLast 5 entries:")
        for i in range(max(0, num_entries - 5), num_entries):
            print(
                f"  Entry {i}: chunk_id={manifest_data[i]['chunk_id']}, num_tokens={manifest_data[i]['num_tokens']}, offset={manifest_data[i]['offset']}"
            )
    print("--- End of Diagnostics ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose an index.bin manifest file.")
    parser.add_argument("manifest_path", type=str, help="Path to the index.bin manifest file.")
    args = parser.parse_args()

    diagnose_manifest_file(args.manifest_path)
