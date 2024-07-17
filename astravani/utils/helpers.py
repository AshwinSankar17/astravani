import json
import random
from typing import List

import torch

WINDOW_FN_SUPPORTED = {
    "hann": torch.hann_window,
    "hamming": torch.hamming_window,
    "blackman": torch.blackman_window,
    "bartlett": torch.bartlett_window,
    "none": None,
}

SUP_DATA_TYPES_SET = {"speaker_id", "pitch", "energy", "reference_audio"}


def read_manifest(path):
    return list(map(json.loads, open(path, "r").readlines()))


def write_manifest(path, manifest, ensure_ascii=False):
    return open(path, "w").writelines(
        [json.dumps(x, ensure_ascii=ensure_ascii) + "\n" for x in manifest]
    )


def update_tracker(tracker, data):
    tracker["data_point"] += 1
    tracker["time"] += data["duration"]


def apply_reduction(losses, reduction="none"):
    """Apply reduction to collection of losses."""
    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    return losses


def stack_tensors(
    tensors: List[torch.Tensor], max_lens: List[int], pad_value: float = 0.0
) -> torch.Tensor:
    """
    Create batch by stacking input tensor list along the time axes.

    Args:
        tensors: List of tensors to pad and stack
        max_lens: List of lengths to pad each axis to, starting with the last axis
        pad_value: Value for padding

    Returns:
        Padded and stacked tensor.
    """
    padded_tensors = []
    for tensor in tensors:
        padding = []
        for i, max_len in enumerate(max_lens, 1):
            padding += [0, max_len - tensor.shape[-i]]

        padded_tensor = torch.nn.functional.pad(tensor, pad=padding, value=pad_value)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.stack(padded_tensors)
    return stacked_tensor


def generate_mask(x, p_cond=0.85, mask_span_length=128):
    """
    Generate a mask for the input tensor x.

    Parameters:
    - x (Tensor): Input tensor of shape (num_frames, num_features).
    - p_cond (float): Probability of applying the mask to a frame.
    - mask_span_length (int): Minimum span length of frames to mask.

    Returns:
    - mask (Tensor): Mask tensor of shape (num_frames, num_features), with 1s indicating masked positions and 0s indicating unmasked positions.
    """
    num_frames, num_features = x.size()
    mask = torch.ones(num_frames, num_features)

    mask_positions = []
    for j in range(num_frames):
        if random.random() < p_cond:
            start_pos = max(0, j - mask_span_length // 2)
            end_pos = min(num_frames, j + mask_span_length // 2)
            mask_positions.extend(range(start_pos, end_pos))

    mask_positions = list(set(mask_positions))  # Remove duplicates
    mask[mask_positions, :] = 0

    return mask
