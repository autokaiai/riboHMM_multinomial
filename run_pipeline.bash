#!/bin/bash

# This script runs six different hyperparameter configurations for the riboHMM pipeline in parallel.
# Each configuration is a combination of a filtering metric (TPM or raw count) and
# the number of top transcripts to keep (500, 1000, or 2000).

# --- Common Configuration ---
BAM_FILES="00_BAMs/uf_muellermcnicoll_2025_04_01_huvec_dnor_2_dedup.bam 00_BAMs/uf_muellermcnicoll_2025_04_02_huvec_dnor_3_dedup.bam 00_BAMs/uf_muellermcnicoll_2025_04_03_huvec_dnor_4_dedup.bam"
FASTA_FILE="01_Reference_Files/MANE.GRCh38.v0.95.select_ensembl_rna.fna"
ORF_ANNOTATIONS="01_Reference_Files/MANE_CDS_coordinates.csv"
MODEL_CONFIG="model_config.json"
RANDOM_STATE=69

# --- Function to run a single pipeline ---
run_pipeline() {
    TOP_N=$1
    METRIC=$2
    SPLIT_METRIC="stratified_${METRIC}"
    ID_TAG="top${TOP_N}_${METRIC}"

    echo "--- Starting pipeline for: ${ID_TAG} with random state ${RANDOM_STATE} ---"

    # Define file paths for this run
    PROCESSED_NPZ="02_Processed_Data/processed_${ID_TAG}.npz"
    PLOT_DIR="03_EDA_Plots/preprocess_${ID_TAG}"
    TRAIN_NPZ="02_Processed_Data/train_${ID_TAG}.npz"
    TEST_NPZ="02_Processed_Data/test_${ID_TAG}.npz"
    MODEL_FILE="04_Models/model_${ID_TAG}.joblib"
    REPORT_FILE="05_Reports/validation_report_${ID_TAG}.pdf"

    # 1. Preprocess Data
    python preprocess.py \
        --bam_files ${BAM_FILES} \
        --fasta_file ${FASTA_FILE} \
        --output_file "${PROCESSED_NPZ}" \
        --plot_dir "${PLOT_DIR}" \
        --filter_top_n "${TOP_N}" \
        --filter_metric "${METRIC}" && \
    
    # 2. Split Data
    python split_data.py \
        --input_npz "${PROCESSED_NPZ}" \
        --output_train_npz "${TRAIN_NPZ}" \
        --output_test_npz "${TEST_NPZ}" \
        --split_method "${SPLIT_METRIC}" \
        --random_state "${RANDOM_STATE}" && \

    # 3. Train Model
    python train.py \
        --input_npz "${TRAIN_NPZ}" \
        --model_config "${MODEL_CONFIG}" \
        --output_model "${MODEL_FILE}" \
        --random_state "${RANDOM_STATE}" && \

    # 4. Validate Model and Generate Report
    python validate.py \
        --input_npz "${TEST_NPZ}" \
        --trained_model "${MODEL_FILE}" \
        --model_config "${MODEL_CONFIG}" \
        --orf_annotations "${ORF_ANNOTATIONS}" \
        --output_report "${REPORT_FILE}"

    echo "--- Finished pipeline for: ${ID_TAG} ---"
}

# --- Run all combinations in parallel ---

# Configurations for TPM
run_pipeline 500 tpm &
run_pipeline 1000 tpm &
run_pipeline 2000 tpm &

# Configurations for raw counts (RPM)
run_pipeline 500 raw &
run_pipeline 1000 raw &
run_pipeline 2000 raw &

# Wait for all parallel jobs to complete
wait
echo "All hyperparameter test runs have completed."