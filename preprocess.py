# Copyright 2025 Kai Alois Wöllstein
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
Pre-processing script for Ribo-Seq Data.

This script serves as the first step in a Ribo-Seq analysis pipeline. It transforms
raw, transcriptome-aligned BAM files into a clean, filtered, and analysis-ready
dataset in .npz format. It also generates several exploratory data analysis (EDA)
plots to provide insights into the data's quality and characteristics.

Core Functionality:
1.  Parses command-line arguments for input files, output files, and filtering criteria.
2.  Analyzes read length distribution across all BAM files to find the dominant length.
3.  Loads transcript lengths from a reference FASTA file, cleaning transcript IDs.
4.  Aggregates 5' read start counts for reads matching the dominant length.
5.  Generates pre-filtering EDA plots to guide filtering decisions.
6.  Filters transcripts based on user-defined criteria (top N or min reads).
7.  Saves the final, filtered dataset to a compressed .npz file.
8.  Generates a `run_summary.json` file detailing all parameters and results.
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import seaborn as sns
from tqdm import tqdm

from utils import calculate_tpm

# --- Helper Function ---
# The calculate_tpm has been moved to utils.py

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the preprocessing script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess Ribo-Seq BAM files into an analysis-ready dataset."
    )
    
    parser.add_argument(
        "--bam_files",
        nargs="+",
        required=True,
        help="One or more paths to the input transcriptome-aligned BAM files.",
    )
    parser.add_argument(
        "--fasta_file",
        type=Path,
        required=True,
        help="Path to the reference FASTA file containing transcript sequences.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Full path for the final, filtered .npz output file.",
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        help="Path to a directory where all generated plots will be saved. "
             "Defaults to the output file's directory.",
    )

    # Mutually exclusive group for filtering
    filter_group = parser.add_mutually_exclusive_group(required=True)
    filter_group.add_argument(
        "--filter_top_n",
        type=int,
        help="Keep only the top N transcripts ranked by total read count.",
    )
    filter_group.add_argument(
        "--filter_min_reads",
        type=int,
        help="Keep only transcripts with at least this minimum total number of reads.",
    )

    parser.add_argument(
        "--filter_metric",
        type=str,
        default="tpm",
        choices=["tpm", "raw"],
        help="Metric for ranking transcripts when using --filter_top_n. "
             "Use 'tpm' (default) for transcript length normalization or "
             "'raw' for raw read counts.",
    )

    parser.add_argument(
        "--no_complex_id_parsing",
        action="store_true",
        help="If set, disable complex ID parsing (e.g., 'ID1|ID2') and use literal BAM transcript IDs.",
    )

    args = parser.parse_args()

    # Post-parsing validation
    if not args.plot_dir:
        args.plot_dir = args.output_file.parent
        logging.info(f"Plot directory not provided. Defaulting to {args.plot_dir}")
    
    # Create output directories if they don't exist
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    for bam_file in args.bam_files:
        if not Path(bam_file).exists():
            logging.error(f"Input BAM file not found: {bam_file}")
            sys.exit(1)

    if not args.fasta_file.exists():
        logging.error(f"Input FASTA file not found: {args.fasta_file}")
        sys.exit(1)

    return args


def get_read_length_distribution(
    bam_files: List[str], plot_dir: Path
) -> Tuple[int, int]:
    """Analyzes read length distribution from BAM files to find the dominant read length.

    Args:
        bam_files: A list of paths to the input BAM files.
        plot_dir: The directory where the read length distribution plot will be saved.

    Returns:
        A tuple containing:
        - dominant_read_length (int): The most frequent read length.
        - median_read_length (int): The median read length.
    """
    logging.info("Analyzing read length distribution from BAM files...")
    read_lengths = Counter()
    total_reads_processed = 0

    for bam_path in bam_files:
        logging.info(f"  -> Processing file: {Path(bam_path).name}")
        with pysam.AlignmentFile(bam_path, "rb") as bamfile:
            for read in tqdm(bamfile.fetch(until_eof=True), desc="Counting reads"):
                total_reads_processed += 1
                if not read.is_unmapped:
                    read_lengths.update([read.query_alignment_length])

    if not read_lengths:
        logging.error("No aligned reads found across all BAM files. Cannot proceed.")
        sys.exit(1)
        
    # Generate Plot 1: Read Length Distribution
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    lengths, counts = zip(*sorted(read_lengths.items()))
    
    ax.bar(lengths, counts, color="skyblue")
    ax.set_title("Read Length Distribution Across All Samples", fontsize=16)
    ax.set_xlabel("Read Length (nt)", fontsize=12)
    ax.set_ylabel("Total Count", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # Set x-axis ticks to be integers
    max_len = max(lengths)
    min_len = min(lengths)
    ax.set_xticks(range(min_len, max_len + 1))
    ax.set_xticklabels(range(min_len, max_len + 1))


    plot_path = plot_dir / "read_length_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"  -> Plot saved to: {plot_path}")

    dominant_read_length, _ = read_lengths.most_common(1)[0]
    logging.info(f"✅ Dominant read length identified as {dominant_read_length} nt.")
    
    return dominant_read_length, total_reads_processed


def load_transcript_lengths(fasta_file: Path) -> Dict[str, int]:
    """Loads transcript lengths from a FASTA file.

    Args:
        fasta_file: Path to the transcript reference FASTA file.

    Returns:
        A dictionary mapping transcript IDs to their lengths.
    """
    logging.info(f"Loading transcript lengths from {fasta_file}...")
    lengths = {}
    
    try:
        # Use pysam.FastaFile for efficient parsing
        with pysam.FastaFile(str(fasta_file)) as fasta:
            for reference_name in tqdm(fasta.references, desc="Parsing FASTA"):
                # Use the full, literal ID from the FASTA header
                clean_id = reference_name
                length = fasta.get_reference_length(reference_name)
                
                if clean_id in lengths:
                     # This can happen if the FASTA contains multiple versions of the same transcript.
                     # We will keep the first one encountered.
                     logging.warning(f"  -> Duplicate clean transcript ID '{clean_id}' found. "
                                     f"Keeping the first one encountered.")
                else:
                    lengths[clean_id] = length
    except Exception as e:
        logging.error(f"Failed to read or parse FASTA file: {e}")
        sys.exit(1)

    if not lengths:
        logging.error("No transcripts could be loaded from the FASTA file. Please check the format.")
        sys.exit(1)

    logging.info(f"✅ Loaded {len(lengths)} transcript lengths from FASTA.")
    return lengths


def aggregate_read_counts(
    bam_files: List[str],
    transcript_lengths: Dict[str, int],
    dominant_read_length: int,
    no_complex_id_parsing: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Aggregates read counts per position for each transcript from BAM files.

    This function iterates through aligned reads in the provided BAM files,
    filters for the dominant read length, and counts the occurrences of the 5'
    end of reads at each nucleotide position of each transcript.

    Args:
        bam_files: A list of paths to input BAM files.
        transcript_lengths: A dictionary mapping transcript IDs to their lengths.
        dominant_read_length: The read length to filter for.
        no_complex_id_parsing: If True, uses a simplified transcript ID parsing logic.

    Returns:
        A tuple containing:
        - A dictionary mapping transcript IDs to a NumPy array of per-position read counts.
        - A dictionary mapping transcript IDs to their total read counts.
    """
    logging.info("Aggregating read counts from BAM files...")

    # Initialize the primary data structure
    aggregated_data = {
        tid: np.zeros(length, dtype=np.int32)
        for tid, length in transcript_lengths.items()
    }
    logging.info(f"  -> Initialized data structure for {len(aggregated_data)} transcripts.")

    unmatched_reads = Counter()

    for bam_path in bam_files:
        logging.info(f"  -> Processing file: {Path(bam_path).name}")
        with pysam.AlignmentFile(bam_path, "rb") as bamfile:
            for read in tqdm(bamfile.fetch(until_eof=True), desc="Aggregating reads"):
                # Filter for dominant read length and mapped reads
                if read.is_unmapped or read.query_alignment_length != dominant_read_length:
                    continue

                raw_tid = read.reference_name
                matched_clean_tid = None

                if no_complex_id_parsing:
                    # Strict, literal matching
                    if raw_tid in aggregated_data:
                        matched_clean_tid = raw_tid
                else:
                    # 1. Try to match the whole complex ID first.
                    if raw_tid in aggregated_data:
                        matched_clean_tid = raw_tid
                    else:
                        # 2. If that fails, iterate over sub-IDs.
                        # Split by both '|' and spaces for robustness.
                        sub_ids = raw_tid.replace(" ", "|").split("|")
                        for sub_id in sub_ids:
                            if sub_id in aggregated_data:
                                matched_clean_tid = sub_id
                                break  # Use the first sub-ID that matches.
                
                # If a match was found, increment the count for the corresponding clean transcript ID.
                if matched_clean_tid:
                    # Ensure position is within bounds
                    if 0 <= read.reference_start < aggregated_data[matched_clean_tid].shape[0]:
                        aggregated_data[matched_clean_tid][read.reference_start] += 1
                else:
                    # If no match was found after all checks, log it as unmatched.
                    unmatched_reads.update([raw_tid])
    
    total_reads_aggregated = sum(np.sum(counts) for counts in aggregated_data.values())
    logging.info(f"✅ Aggregation complete. Total of {total_reads_aggregated:,} reads were aggregated.")
    
    if unmatched_reads:
        total_unmatched_count = sum(unmatched_reads.values())
        logging.warning(
            f"  -> Found {total_unmatched_count:,} reads mapping to "
            f"{len(unmatched_reads)} transcript IDs not present in the FASTA file."
        )

    return aggregated_data, unmatched_reads


def generate_pre_filtering_plots(
    aggregated_data: Dict[str, np.ndarray],
    transcript_lengths: Dict[str, int],
    plot_dir: Path,
) -> None:
    """Generates and saves plots for data analysis before filtering.

    This includes a plot of total read counts per transcript ranked by
    abundance and a plot of the distribution of transcript lengths.

    Args:
        aggregated_data: Dictionary mapping transcript IDs to read count arrays.
        transcript_lengths: Dictionary mapping transcript IDs to their lengths.
        plot_dir: Directory to save the plots.
    """
    logging.info("Generating pre-filtering EDA plots...")

    total_reads_per_transcript = {
        tid: np.sum(counts) for tid, counts in aggregated_data.items()
    }
    
    if not total_reads_per_transcript:
        logging.warning("No reads were aggregated. Skipping pre-filtering plots.")
        return

    # Calculate TPM to rank by read density
    tpm_per_transcript = calculate_tpm(total_reads_per_transcript, transcript_lengths)

    # Create a DataFrame for easy sorting and manipulation
    total_df = pd.DataFrame.from_dict(
        tpm_per_transcript, orient="index", columns=["tpm"]
    )
    sorted_df = total_df.sort_values(by="tpm", ascending=False)

    # Define the "top N" categories for the boxplot
    tiers = [100, 500, 1000, 2000, 3000, 4000, 5000, 10000]
    plot_data = []

    for n in tiers:
        # Ensure we don't try to select more transcripts than exist
        if len(sorted_df) >= n:
            top_n_df = sorted_df.head(n).copy()
            top_n_df["Category"] = f"Top {n}"
            plot_data.append(top_n_df)
    
    if not plot_data:
        logging.warning("Not enough transcripts to generate the ranked distribution plot. Skipping.")
        return

    plot_df = pd.concat(plot_data)

    # Create the boxplot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 9))

    sns.boxplot(data=plot_df, x="Category", y="tpm", ax=ax)

    ax.set_yscale("log")
    ax.set_title("Distribution of TPM per Transcript by Rank", fontsize=18)
    ax.set_xlabel("Transcript Rank Group (by TPM)", fontsize=14)
    ax.set_ylabel("Transcripts Per Million (TPM) (log scale)", fontsize=14)
    ax.tick_params(axis="x", rotation=45)

    # Add annotations for key percentiles
    for i, n in enumerate(tiers):
        if len(sorted_df) >= n:
            category_data = plot_df[plot_df["Category"] == f"Top {n}"]["tpm"]
            percentiles = category_data.quantile([0, 0.25, 0.5, 0.75, 1.0]).to_dict()
            
            # Position the annotation text box
            x_pos = i
            y_pos = category_data.median()
            
            annotation_text = (
                f"Max: {percentiles[1.0]:.2f}\n"
                f"75th: {percentiles[0.75]:.2f}\n"
                f"Median: {percentiles[0.5]:.2f}\n"
                f"25th: {percentiles[0.25]:.2f}\n"
                f"Min: {percentiles[0.0]:.2f}"
            )

            ax.text(x_pos, y_pos, annotation_text,
                    ha='center', va='center', fontsize=9, color='black',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))


    plot_path = plot_dir / "read_count_distribution_by_rank.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"  -> Plot saved to: {plot_path}")
    logging.info("✅ Pre-filtering plots generated.")


def filter_transcripts(
    aggregated_data: Dict[str, np.ndarray],
    transcript_lengths: Dict[str, int],
    filter_top_n: Optional[int],
    filter_min_reads: Optional[int],
    filter_metric: str = "tpm",
) -> Dict[str, np.ndarray]:
    """Filters transcripts based on expression levels or read counts.

    Args:
        aggregated_data: Dictionary mapping transcript IDs to read count arrays.
        transcript_lengths: Dictionary mapping transcript IDs to lengths.
        filter_top_n: If set, keeps the top N transcripts by the specified metric.
        filter_min_reads: If set, keeps transcripts with at least this many total reads.
        filter_metric: The metric for ranking ('tpm' or 'raw'). Defaults to 'tpm'.

    Returns:
        A dictionary containing the filtered transcript data.
    """
    logging.info("Filtering transcripts...")
    
    total_transcripts_before = len(aggregated_data)
    if total_transcripts_before == 0:
        logging.warning("Aggregated data is empty. No filtering will be performed.")
        return {}
        
    total_reads_per_transcript = {
        tid: np.sum(counts) for tid, counts in aggregated_data.items()
    }

    filtered_ids = set()

    if filter_top_n is not None:
        logging.info(f"  -> Applying filter: Keep top {filter_top_n} transcripts by {filter_metric}.")
        
        values_to_sort = {}
        if filter_metric == "tpm":
            values_to_sort = calculate_tpm(total_reads_per_transcript, transcript_lengths)
        else: # 'raw'
            values_to_sort = total_reads_per_transcript

        if filter_top_n > total_transcripts_before:
            logging.warning(
                f"  -> Requested top {filter_top_n} transcripts, but only "
                f"{total_transcripts_before} are available. Keeping all."
            )
            filter_top_n = total_transcripts_before
            
        sorted_transcripts = sorted(
            values_to_sort.items(), key=lambda item: item[1], reverse=True
        )
        filtered_ids = {tid for tid, count in sorted_transcripts[:filter_top_n]}

    elif filter_min_reads is not None:
        logging.info(f"  -> Applying filter: Keep transcripts with >= {filter_min_reads} reads.")
        filtered_ids = {
            tid for tid, count in total_reads_per_transcript.items() 
            if count >= filter_min_reads
        }
    
    final_data = {
        tid: aggregated_data[tid] for tid in filtered_ids
    }

    logging.info(
        f"✅ Filtering complete. Retained {len(final_data)} out of "
        f"{total_transcripts_before} total transcripts."
    )
    
    return final_data


def save_final_output(
    filtered_data: Dict[str, np.ndarray],
    output_file: Path,
) -> None:
    """Saves the processed transcript data to a compressed NPZ file.

    Args:
        filtered_data: The final dictionary of transcript data to save.
        output_file: The path to the output .npz file.
    """
    logging.info(f"Saving final processed data to {output_file}...")

    if not filtered_data:
        logging.warning("Filtered data is empty. An empty .npz file will be created.")

    try:
        np.savez_compressed(output_file, **filtered_data)
        logging.info("✅ Final output saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save the final output file: {e}")
        sys.exit(1)


def save_unmatched_ids_summary(unmatched_data: Counter, plot_dir: Path) -> None:
    """Saves a summary of transcript IDs found in BAM files but not in the reference FASTA.

    Args:
        unmatched_data: A Counter object with unmatched transcript IDs and their counts.
        plot_dir: The directory to save the summary file.
    """
    if not unmatched_data:
        logging.info("No unmatched IDs to report. Skipping summary file.")
        return

    summary_file_path = plot_dir / "unmatched_ids_summary.json"
    logging.info(f"Saving summary of unmatched transcript IDs to {summary_file_path}...")
    
    # Sort the unmatched IDs by read count in descending order.
    # The `most_common()` method of Counter is perfect for this.
    sorted_unmatched_items = unmatched_data.most_common()

    # Format the data as requested by the user, preserving the sorted order.
    output_dict = {
        complex_id: {
            "read_count": count,
            "full_header": complex_id,
        }
        for complex_id, count in sorted_unmatched_items
    }

    try:
        with open(summary_file_path, 'w') as f:
            json.dump(output_dict, f, indent=4)
        logging.info("✅ Unmatched IDs summary saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save the unmatched IDs summary file: {e}")


def save_run_summary(summary_data: Dict, plot_dir: Path) -> None:
    """Saves a JSON summary of the preprocessing run.

    Args:
        summary_data: A dictionary containing key metrics and configuration of the run.
        plot_dir: The directory where the summary will be saved.
    """
    summary_file_path = plot_dir / "run_summary.json"
    logging.info(f"Saving run summary to {summary_file_path}...")
    try:
        with open(summary_file_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        logging.info("✅ Run summary saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save the run summary file: {e}")
        # We don't exit here, as this is a non-critical file.


def main():
    """Main function to orchestrate the preprocessing workflow."""
    args = parse_arguments()

    # --- Data for the final run summary ---
    run_summary = {
        "command": " ".join(sys.argv),
        "run_timestamp": datetime.now().isoformat(),
        "input_bam_files": [str(Path(p).resolve()) for p in args.bam_files],
        "input_fasta_file": str(args.fasta_file.resolve()),
        "output_npz_file": str(args.output_file.resolve()),
        "plot_directory": str(args.plot_dir.resolve()),
    }

    # Step 1: Analyze read length distribution and get the dominant length
    dominant_read_length, total_reads_processed = get_read_length_distribution(
        args.bam_files, args.plot_dir
    )
    run_summary["dominant_read_length"] = dominant_read_length
    run_summary["total_reads_processed"] = total_reads_processed

    # Step 2: Load transcript lengths from the FASTA file
    transcript_lengths = load_transcript_lengths(args.fasta_file)
    run_summary["total_transcripts_in_fasta"] = len(transcript_lengths)

    # Step 3: Aggregate read counts from all BAM files
    aggregated_data, unmatched_reads = aggregate_read_counts(
        args.bam_files,
        transcript_lengths,
        dominant_read_length,
        args.no_complex_id_parsing,
    )
    run_summary["transcripts_before_filtering"] = len(aggregated_data)
    run_summary["total_reads_aggregated_to_transcripts"] = int(sum(np.sum(c) for c in aggregated_data.values()))
    run_summary["unmatched_transcript_ids"] = {
        "unique_unmatched_ids": len(unmatched_reads),
        "total_unmatched_reads": sum(unmatched_reads.values())
    }


    # Step 4: Generate pre-filtering EDA plots based on TPM
    generate_pre_filtering_plots(aggregated_data, transcript_lengths, args.plot_dir)

    # Step 5: Filter the data based on user criteria (now uses TPM for top_n)
    filtered_data = filter_transcripts(
        aggregated_data,
        transcript_lengths,
        args.filter_top_n,
        args.filter_min_reads,
        args.filter_metric,
    )
    run_summary["transcripts_after_filtering"] = len(filtered_data)
    
    if args.filter_top_n:
        run_summary["filter_used"] = {
            "method": "top_n",
            "value": args.filter_top_n,
            "metric": args.filter_metric,
        }
    else:
        run_summary["filter_used"] = {
            "method": "min_reads", "value": args.filter_min_reads
        }


    # Step 6: Save the final, filtered data and the summary
    save_final_output(filtered_data, args.output_file)
    save_run_summary(run_summary, args.plot_dir)
    save_unmatched_ids_summary(unmatched_reads, args.plot_dir)

    logging.info("✅ Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main() 