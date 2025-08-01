import numpy as np
import pytest
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from split_data import split_transcript_ids

@pytest.fixture
def sample_transcript_data():
    """Create a sample transcript dataset for testing."""
    return {
        f"T{i}": np.random.randint(0, 100, size=np.random.randint(50, 200))
        for i in range(100)
    }

def test_random_split_reproducibility(sample_transcript_data):
    """Test that the random split is reproducible with the same random state."""
    all_tids = list(sample_transcript_data.keys())
    
    # Pass a copy of the list to the function to avoid in-place modification issues
    train1, test1 = split_transcript_ids(
        all_tids.copy(), sample_transcript_data, 'random', 0.8, 42, 4
    )
    train2, test2 = split_transcript_ids(
        all_tids.copy(), sample_transcript_data, 'random', 0.8, 42, 4
    )
    
    assert set(train1) == set(train2)
    assert set(test1) == set(test2)

def test_stratified_split_proportions(sample_transcript_data):
    """Test that stratified sampling maintains the correct proportions."""
    all_tids = list(sample_transcript_data.keys())
    
    train_ids, test_ids = split_transcript_ids(
        all_tids, sample_transcript_data, 'stratified_raw', 0.8, 42, 4
    )
    
    assert len(train_ids) + len(test_ids) == 100
    # Check if the split is approximately 80/20
    assert 75 < len(train_ids) < 85
    assert 15 < len(test_ids) < 25
    # Ensure there's no overlap
    assert len(set(train_ids).intersection(set(test_ids))) == 0 