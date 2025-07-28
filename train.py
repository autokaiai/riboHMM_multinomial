"""
Train a riboHMM Model.

This script serves as a lean, focused command-line tool to train a Hidden Markov
Model (HMM) for Ribo-Seq data analysis. Its sole responsibility is to take
preprocessed training data and a model configuration file, train the HMM, and
save the resulting model object for later use in validation or prediction.

Core Functionality:
1.  Parses command-line arguments for input data, model configuration, output
    path, and training hyperparameters.
2.  Loads all architectural and initial model parameters from a central JSON
    configuration file, including state maps, transition/emission matrices,
    and windowing parameters.
3.  Prepares HMM observation sequences (lagging windows) for all transcripts
    in the provided dataset.
4.  Initializes an hmmlearn.MultinomialHMM model with the specified parameters.
5.  Trains the model on the entire dataset.
6.  Saves the final, trained model object using joblib.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import joblib
import numpy as np
from hmmlearn import hmm
from tqdm import tqdm

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a Hidden Markov Model (HMM) on ribosome profiling data."
    )
    parser.add_argument(
        "--input_npz",
        type=Path,
        required=True,
        help="Path to the .npz file containing the training data.",
    )
    parser.add_argument(
        "--model_config",
        type=Path,
        required=True,
        help="Path to the JSON configuration file for the model.",
    )
    parser.add_argument(
        "--output_model",
        type=Path,
        required=True,
        help="Path to save the trained model (e.g., riboHMM.joblib).",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=100,
        help="The maximum number of training iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="The convergence threshold for training.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="The random seed for reproducibility.",
    )

    args = parser.parse_args()
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    return args


def load_data(
    npz_path: Path, config_path: Path
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Loads training data and model configuration.

    Args:
        npz_path: Path to the NPZ file containing transcript read counts.
        config_path: Path to the JSON file with model configuration.

    Returns:
        A tuple containing:
        - A dictionary with transcript data.
        - A dictionary with the model configuration.
    """
    logging.info(f"Loading data from {npz_path} and config from {config_path}...")
    try:
        data_loader = np.load(npz_path, allow_pickle=True)
        # Exclude metadata key from transcript data
        transcript_data = {
            key: data_loader[key]
            for key in data_loader.files
        }
        logging.info(f"  -> Loaded {len(transcript_data)} transcripts from {npz_path}")
    except Exception as e:
        logging.error(f"Failed to load NPZ file: {e}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            model_config = json.load(f)
        logging.info(f"  -> Loaded model configuration from {config_path}")
    except Exception as e:
        logging.error(f"Failed to load model config file: {e}")
        sys.exit(1)

    return transcript_data, model_config


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


def initialize_model(
    model_config: Dict[str, Any], n_iter: int, tol: float, random_state: int
) -> hmm.MultinomialHMM:
    """Initializes the Multinomial HMM with parameters from the config.

    Args:
        model_config: Dictionary containing model parameters.
        n_iter: Maximum number of iterations for the Baum-Welch algorithm.
        tol: Convergence threshold for the Baum-Welch algorithm.
        random_state: Seed for the random number generator.

    Returns:
        An initialized (but not yet trained) MultinomialHMM object.
    """
    state_map = model_config["state_map"]
    n_components = len(state_map)
    window_size = model_config["WINDOW_SIZE"]
    state_labels = list(state_map.keys())

    initial_transmat = np.array(model_config["initial_transition_matrix"])

    emission_blueprints = model_config["emission_blueprints"]
    unnormalized_emissions = np.array(
        [emission_blueprints[label] for label in state_labels]
    )
    initial_emissions = normalize_emissions(unnormalized_emissions)

    model = hmm.MultinomialHMM(
        n_components=n_components,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        params="te",  # Train transitions and emissions
        init_params="",  # Do not initialize params randomly, use blueprints
    )

    model.n_features = window_size
    model.startprob_ = np.array([1.0] + [0.0] * (n_components - 1))
    model.transmat_ = initial_transmat
    model.emissionprob_ = initial_emissions

    return model


def main():
    """Main function to orchestrate the HMM training process."""
    args = parse_arguments()
    transcript_data, model_config = load_data(args.input_npz, args.model_config)

    window_size = model_config["WINDOW_SIZE"]
    lag_offset = model_config["LAG_OFFSET"]

    # Prepare training data from all transcripts
    logging.info("Preparing training data (lagging windows)...")
    X_list = []
    lengths = []
    for tid, counts in tqdm(transcript_data.items(), desc="Processing transcripts"):
        read_counts = counts + model_config["PSEUDO_COUNT_TRAIN"]
        X_single = create_lagging_windows(read_counts, window_size, lag_offset)
        if X_single.shape[0] > 0:
            X_list.append(X_single)
            lengths.append(X_single.shape[0])

    if not X_list:
        logging.error("No valid training data could be generated. Aborting.")
        sys.exit(1)

    X_train = np.vstack(X_list).astype(np.int32)

    logging.info("Starting HMM training...")
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Number of sequences: {len(lengths)}")
    model = initialize_model(
        model_config, args.n_iter, args.tol, args.random_state
    )
    model.fit(X_train, lengths)
    logging.info("✅ HMM training completed.")

    logging.info(f"Saving trained model to {args.output_model}...")
    try:
        joblib.dump(model, args.output_model)
        logging.info("✅ Model saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 