"""
mask_utils.py
Utilities for encoding/decoding binary masks.

Uses Run-Length Encoding (RLE) for efficient storage.
"""

import numpy as np
import base64
import zlib


def encode_mask_rle(mask: np.ndarray) -> str:
    """
    Encode binary mask as compressed RLE string.

    Args:
        mask: Binary mask array (H, W) with dtype bool or uint8

    Returns:
        Base64-encoded compressed RLE string
    """
    # Flatten and ensure binary
    flat = mask.flatten().astype(np.uint8)

    # RLE encoding: store (value, count) pairs
    # But simpler: just store the positions where values change
    diff = np.diff(flat)
    change_indices = np.where(diff != 0)[0] + 1

    # Start positions of each run
    starts = np.concatenate([[0], change_indices])

    # Store: first_value, then run lengths
    first_val = flat[0]
    run_lengths = np.diff(np.concatenate([starts, [len(flat)]]))

    # Pack: first byte is first_value, then run lengths as uint16
    data = np.concatenate([[first_val], run_lengths]).astype(np.uint32)

    # Also store shape for decoding
    shape_data = np.array(mask.shape, dtype=np.uint32)
    full_data = np.concatenate([shape_data, data])

    # Compress and encode
    compressed = zlib.compress(full_data.tobytes())
    return base64.b64encode(compressed).decode('ascii')


def decode_mask_rle(rle_string: str) -> np.ndarray:
    """
    Decode RLE string back to binary mask.

    Args:
        rle_string: Base64-encoded compressed RLE string

    Returns:
        Binary mask array (H, W)
    """
    # Decode and decompress
    compressed = base64.b64decode(rle_string.encode('ascii'))
    data = np.frombuffer(zlib.decompress(compressed), dtype=np.uint32)

    # Extract shape and RLE data
    h, w = data[0], data[1]
    first_val = data[2]
    run_lengths = data[3:]

    # Reconstruct mask
    flat = np.zeros(h * w, dtype=np.uint8)
    current_val = first_val
    pos = 0

    for length in run_lengths:
        flat[pos:pos + length] = current_val
        pos += length
        current_val = 1 - current_val  # Toggle 0/1

    return flat.reshape(h, w).astype(bool)


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    # Test with a simple mask
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 30:60] = True  # Rectangle
    mask[50:70, 10:30] = True  # Another rectangle

    # Encode
    rle = encode_mask_rle(mask)
    print(f"Original mask: {mask.shape}, {mask.sum()} pixels")
    print(f"RLE length: {len(rle)} chars")

    # Decode
    decoded = decode_mask_rle(rle)
    print(f"Decoded mask: {decoded.shape}, {decoded.sum()} pixels")
    print(f"Match: {np.array_equal(mask, decoded)}")
