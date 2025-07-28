# riboHMM: A Hidden Markov Model for Ribosome Profiling Data

This repository contains the code for `riboHMM`, a Hidden Markov Model (HMM) designed to analyze ribosome profiling (Ribo-Seq) data. The model identifies translational states (e.g., 5'UTR, initiation, elongation, termination, 3'UTR) based on the density of ribosome footprints along transcripts.

## Project Structure

- `preprocess.py`: Scripts for processing raw BAM files into transcript-level read count arrays.
- `split_data.py`: Splits the processed data into training and testing sets for model development.
- `train.py`: Trains the HMM on the training dataset.
- `validate.py`: Evaluates the trained HMM on the test set and generates a validation report with performance metrics and visualizations.
- `run_pipeline.bash`: A bash script that orchestrates the entire workflow from preprocessing to validation.
- `model_config.json`: A configuration file specifying the HMM architecture, state definitions, and initial parameters.

## Getting Started

### Prerequisites

- Python 3.8+
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is recommended for environment management.
- Required Python packages can be installed via a requirements file (not included yet, but recommended). Key packages include:
  - `hmmlearn`
  - `numpy`
  - `pandas`
  - `pysam`
  - `matplotlib`
  - `seaborn`
  - `joblib`
  - `tqdm`
  - `fpdf`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd riboHMM_simple_prod
    ```

2.  **Set up the environment (recommended):**
    ```bash
    conda create --name ribohmm python=3.9
    conda activate ribohmm
    # pip install -r requirements.txt 
    ```

## Usage

The entire pipeline can be executed using the `run_pipeline.bash` script. This script handles preprocessing, data splitting, training, and validation in a single command.

### Example Command

Here is an example of how to run the full pipeline. You will need to provide paths to your reference files and BAM files.

```bash
bash run_pipeline.bash \
    --fasta-file 01_Reference_Files/MANE.GRCh38.v0.95.select_ensembl_rna.fna \
    --orf-annotations 01_Reference_Files/MANE_CDS_coordinates.csv \
    --bam-files "00_BAMs/sample1.bam 00_BAMs/sample2.bam" \
    --top-n 1000 \
    --model-suffix "top1000_tpm"
```

### Script Arguments

- `--fasta-file`: Path to the reference transcriptome in FASTA format.
- `--orf-annotations`: Path to a CSV file with ORF start and end coordinates.
- `--bam-files`: A space-separated list of input BAM files, enclosed in quotes.
- `--top-n`: The number of top-expressed transcripts to use for training and validation (e.g., 1000).
- `--model-suffix`: A unique suffix to append to all output files (e.g., "top1000_tpm"), helping to organize results from different runs.

### Pipeline Steps

1.  **Preprocessing (`preprocess.py`):**
    - Aggregates 5' read counts from BAM files for each transcript.
    - Filters for the top N most expressed transcripts based on TPM or raw counts.
    - Saves the processed data as an `.npz` file.

2.  **Data Splitting (`split_data.py`):**
    - Splits the processed transcripts into stratified training and test sets.
    - Saves two `.npz` files (`train_set_...` and `test_set_...`).

3.  **Training (`train.py`):**
    - Trains the HMM using the training set.
    - Saves the trained model as a `.joblib` file.

4.  **Validation (`validate.py`):**
    - Evaluates the model on the test set.
    - Generates a comprehensive PDF report (`validation_report_...`) with visualizations of model parameters, state predictions, and quantitative metrics.

This workflow ensures that the model is trained and evaluated systematically, with all artifacts clearly named and organized. 