# Copyright 2025 Kai Wöllstein, Marcel Schulz, Christina Kalk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for the riboHMM pipeline.
"""
import numpy as np
from typing import Dict


def calculate_tpm(
    read_counts: Dict[str, int], transcript_lengths: Dict[str, int]
) -> Dict[str, float]:
    """Calculates Transcripts Per Million (TPM) for a set of transcripts.

    Args:
        read_counts: A dictionary mapping transcript IDs to their total read counts.
        transcript_lengths: A dictionary mapping transcript IDs to their lengths in nucleotides.

    Returns:
        A dictionary mapping transcript IDs to their TPM values.
    """
    rpk_values = {}
    
    # Calculate RPK (Reads Per Kilobase)
    for tid, count in read_counts.items():
        length_kb = transcript_lengths.get(tid, 0) / 1000.0
        if length_kb > 0:
            rpk_values[tid] = count / length_kb
        else:
            rpk_values[tid] = 0

    # Calculate the "per million" scaling factor
    total_rpk = sum(rpk_values.values())
    if total_rpk == 0:
        return {tid: 0.0 for tid in read_counts.keys()}
    
    scaling_factor = total_rpk / 1_000_000.0

    # Calculate TPM for each transcript
    tpm_values = {tid: rpk / scaling_factor for tid, rpk in rpk_values.items()}
    return tpm_values


def create_lagging_windows(
    read_counts: np.ndarray, window_size: int, lag_offset: int, constant_values: int = 1
) -> np.ndarray:
    """Creates lagging windows from a 1D array of read counts.

    This function generates a 2D array of sliding windows over the input read
    counts. The `lag_offset` parameter allows the windows to be shifted relative
    to the start of the sequence, which is useful for aligning features like
    the P-site of a ribosome.

    Args:
        read_counts: A 1D NumPy array of read counts for a single transcript.
        window_size: The size of each sliding window.
        lag_offset: The number of positions to shift the window start. This is
                    achieved by prepending `lag_offset` constant values.
        constant_values: The value to use for padding. Defaults to 1.

    Returns:
        A 2D NumPy array where each row is a window of size `window_size`.
    """
    padded_counts = np.pad(
        read_counts, (lag_offset, 0), "constant", constant_values=constant_values
    )
    num_windows = len(read_counts)
    if num_windows == 0:
        return np.empty((0, window_size), dtype=np.int32)
    shape = (num_windows, window_size)
    strides = (padded_counts.strides[0], padded_counts.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        padded_counts, shape=shape, strides=strides
    )
    return windows


def normalize_emissions(unnormalized_emissions: np.ndarray) -> np.ndarray:
    """Normalizes emission probabilities to ensure they sum to 1 across each state.

    Args:
        unnormalized_emissions: A 2D NumPy array of emission blueprints.

    Returns:
        A 2D NumPy array with normalized emission probabilities.
    """
    row_sums = unnormalized_emissions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return unnormalized_emissions / row_sums 