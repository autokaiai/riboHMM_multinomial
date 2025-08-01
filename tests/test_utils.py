import numpy as np
import pytest
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import create_lagging_windows, calculate_tpm, normalize_emissions

def test_create_lagging_windows_basic():
    """Test the basic functionality of create_lagging_windows."""
    read_counts = np.array([1, 2, 3, 4, 5])
    windows = create_lagging_windows(read_counts, window_size=3, lag_offset=2, constant_values=0)
    
    assert windows.shape == (5, 3), "Output shape is incorrect"
    
    expected_windows = np.array([
        [0, 0, 1],
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ])
    
    np.testing.assert_array_equal(windows, expected_windows, "Window contents are not as expected")

def test_create_lagging_windows_empty_input():
    """Test create_lagging_windows with empty input."""
    read_counts = np.array([])
    windows = create_lagging_windows(read_counts, window_size=3, lag_offset=2)
    assert windows.shape == (0, 3), "Should return an empty array with the correct number of columns"

def test_calculate_tpm_basic():
    """Test basic TPM calculation."""
    read_counts = {"t1": 10, "t2": 20, "t3": 5}
    transcript_lengths = {"t1": 1000, "t2": 2000, "t3": 500}
    tpm = calculate_tpm(read_counts, transcript_lengths)
    
    # RPKs: t1=10, t2=10, t3=10. Total RPK = 30.
    # Scaling factor = 30 / 1e6 = 3e-5
    # TPMs: t1=10/3e-5, etc. All should be equal.
    assert np.isclose(tpm["t1"], tpm["t2"])
    assert np.isclose(tpm["t2"], tpm["t3"])
    assert np.isclose(sum(tpm.values()), 1_000_000)

def test_normalize_emissions_basic():
    """Test that emission probabilities are correctly normalized."""
    unnormalized = np.array([
        [10, 20, 70],
        [0, 0, 0],
        [5, 5, 5]
    ])
    normalized = normalize_emissions(unnormalized)
    
    # Check that rows sum to 1
    row_sums = normalized.sum(axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 0.0, 1.0], err_msg="Rows should sum to 1, or 0 if all inputs were 0")
    
    # Check a specific row's values
    np.testing.assert_allclose(normalized[0], [0.1, 0.2, 0.7]) 