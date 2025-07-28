"""
Split Preprocessed Ribo-Seq Data into Training and Test Sets.

This script serves as a dedicated utility to partition a preprocessed Ribo-Seq
dataset (in .npz format) into two disjoint sets: one for training a model and
one for testing/validating it.

By creating a reproducible, file-based split, it ensures that the model is
never validated on data it has been trained on, which is a critical step for
obtaining meaningful performance metrics.

The script supports two splitting strategies:
1.  **Random**: A simple, random shuffling of all transcripts (default).
2.  **Stratified**: Stratified sampling based on transcript expression levels
    (either raw counts or TPM), ensuring both training and test sets have a
    similar distribution of low, medium, and high-expression transcripts.

Core Functionality:
1.  Parses command-line arguments for the input dataset, output paths, and
    the desired splitting strategy.
2.  Loads the full dataset of transcript read counts.
3.  Splits transcript IDs based on the chosen method.
4.  Saves the corresponding data into two separate .npz files, one for training
    and one for testing.
"""
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# --- Helper Function for TPM Calculation ---
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
    rpk = {
        tid: (reads / (transcript_lengths[tid] / 1000.0))
        for tid, reads in read_counts.items()
    }

    # Calculate the "per million" scaling factor
    total_rpk = sum(rpk.values())
    if total_rpk == 0:
        return {tid: 0.0 for tid in read_counts.keys()}
    
    scaling_factor = total_rpk / 1_000_000.0

    # Calculate TPM for each transcript
    tpm_values = {tid: rpk_val / scaling_factor for tid, rpk_val in rpk.items()}
    return tpm_values


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the data splitting script."""
    parser = argparse.ArgumentParser(
        description="Split transcript data into training and testing sets."
    )
    parser.add_argument(
        "--input_npz",
        type=Path,
        required=True,
        help="Path to the full .npz dataset from preprocess.py.",
    )
    parser.add_argument(
        "--output_train_npz",
        type=Path,
        required=True,
        help="Path to save the training data subset (.npz).",
    )
    parser.add_argument(
        "--output_test_npz",
        type=Path,
        required=True,
        help="Path to save the test data subset (.npz).",
    )
    parser.add_argument(
        "--train_split_fraction",
        type=float,
        default=0.8,
        help="The fraction of transcripts to allocate to the training set.",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="random",
        choices=["random", "stratified_raw", "stratified_tpm"],
        help="Method for splitting data: 'random' (default shuffle), "
             "'stratified_raw' (stratified sampling on raw counts), or "
             "'stratified_tpm' (stratified sampling on TPM).",
    )
    parser.add_argument(
        "--stratification_bins",
        type=int,
        default=4,
        help="Number of bins (quantiles) for stratified sampling. Default is 4 (quartiles).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility of the split.",
    )

    args = parser.parse_args()

    # Create output directories if they don't exist
    args.output_train_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_test_npz.parent.mkdir(parents=True, exist_ok=True)

    return args


def main():
    """Main function to load, stratify, split, and save the data."""
    args = parse_arguments()

    if not args.input_npz.exists():
        logging.error(f"Input file not found: {args.input_npz}")
        sys.exit(1)

    logging.info(f"Loading data from {args.input_npz}...")
    try:
        data_loader = np.load(args.input_npz, allow_pickle=True)
        transcript_data = {
            key: data_loader[key]
            for key in data_loader.files
        }
    except Exception as e:
        logging.error(f"Failed to load and parse NPZ file: {e}")
        sys.exit(1)

    all_tids = list(transcript_data.keys())
    logging.info(f"Loaded {len(all_tids)} total transcripts.")

    # --- Splitting logic ---
    if args.split_method == "random":
        logging.info("Splitting data randomly...")
        # Shuffle the transcript IDs for a random split
        random.Random(args.random_state).shuffle(all_tids)
        # Determine the split index
        split_idx = int(len(all_tids) * args.train_split_fraction)
        train_set_ids = all_tids[:split_idx]
        test_set_ids = all_tids[split_idx:]
    else: # Stratified sampling
        metric_name = "raw counts" if "raw" in args.split_method else "TPM"
        logging.info(
            f"Performing stratified split based on {metric_name} "
            f"using {args.stratification_bins} bins..."
        )
        
        raw_counts = {
            tid: np.sum(counts) for tid, counts in transcript_data.items()
        }

        if "tpm" in args.split_method:
            transcript_lengths = {
                tid: len(counts) for tid, counts in transcript_data.items()
            }
            metric_values = calculate_tpm(raw_counts, transcript_lengths)
        else: # raw counts
            metric_values = raw_counts

        # Create a Series for easy binning
        metrics_series = pd.Series(metric_values)
        
        # Bin transcripts into quantiles, drop bins with no unique edges
        try:
            bins = pd.qcut(
                metrics_series, q=args.stratification_bins, labels=False, duplicates='drop'
            )
        except ValueError as e:
            logging.error(f"Could not create {args.stratification_bins} bins. "
                          f"Try fewer bins. Error: {e}")
            sys.exit(1)
        
        train_set_ids = []
        test_set_ids = []

        # Group transcript IDs by their assigned bin
        binned_tids = pd.Series(
            metrics_series.index, index=metrics_series.index
        ).groupby(bins)

        for bin_label, tids_in_bin in binned_tids:
            tids_list = tids_in_bin.tolist()
            # Shuffle within the bin for randomness
            random.Random(args.random_state).shuffle(tids_list)
            
            split_idx = round(len(tids_list) * args.train_split_fraction)
            train_set_ids.extend(tids_list[:split_idx])
            test_set_ids.extend(tids_list[split_idx:])

        # Final shuffle of the collected IDs to mix bins
        random.Random(args.random_state).shuffle(train_set_ids)
        random.Random(args.random_state).shuffle(test_set_ids)
    
    logging.info(
        f"Splitting data: {len(train_set_ids)} for training, {len(test_set_ids)} for testing."
    )

    # Create the data dictionaries for each set
    train_data = {
        tid: transcript_data[tid]
        for tid in tqdm(train_set_ids, desc="Creating train set")
    }
    test_data = {
        tid: transcript_data[tid] for tid in tqdm(test_set_ids, desc="Creating test set")
    }

    # Save the new .npz files
    logging.info(f"Saving training set to {args.output_train_npz}...")
    try:
        np.savez_compressed(args.output_train_npz, **train_data)
    except Exception as e:
        logging.error(f"Failed to save training NPZ file: {e}")
        sys.exit(1)

    logging.info(f"Saving test set to {args.output_test_npz}...")
    try:
        np.savez_compressed(args.output_test_npz, **test_data)
    except Exception as e:
        logging.error(f"Failed to save test NPZ file: {e}")
        sys.exit(1)

    logging.info("✅ Data splitting completed successfully!")


if __name__ == "__main__":
    main() 