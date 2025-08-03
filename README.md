# `riboHMM2.0`: A Modern Hidden Markov Model for Accurate Reading Frame Identification in Ribosome Profiling Data

This repository contains `riboHMM2.0`, a modernized and production-ready pipeline for analyzing Ribosome Profiling (Ribo-Seq) data. This project builds upon the foundational ideas of the original `riboHMM` developed by Raj et al. (2016), which can be found on [GitHub here](https://github.com/rajanil/riboHMM).

Our work introduces a novel approach to modeling the observational data, simplifying the architecture while retaining high performance in reading frame identification.

This README serves as a comprehensive guide and scientific summary of the project, intended for a broad audience.

A project by the **Institute for Computational Medicine, Goethe University Frankfurt**.

**Supervisors:**
- Prof. Dr. Marcel Schulz
- Christina Kalk

---

## Table of Contents
- [Scientific Background](#scientific-background)
- [Our Approach: A "Weather Detective" for the Genome](#our-approach-a-weather-detective-for-the-genome)
- [Key Results: What We Found](#key-results-what-we-found)
- [The `riboHMM2.0` Pipeline: How It Works](#the-ribohmm20-pipeline-how-it-works)
- [Usage Guide (For Technical Users)](#usage-guide-for-technical-users)
- [Project Structure](#project-structure)
- [Conclusion & Future Work](#conclusion--future-work)

---

## Scientific Background

### The Challenge: Which Genes are "On"?
The central dogma of molecular biology describes how the genetic information in **DNA** is transcribed into **RNA**, which is then translated into **proteins** (the building blocks and workhorses of our cells). While modern technology has made it easy to sequence DNA and RNA, a fundamental challenge remains: how do we know which specific parts of an RNA molecule are actually being used to create proteins at any given moment? Answering this question is crucial for understanding health, disease, and the fundamental workings of life.

### Ribosome Profiling: A Snapshot of Cellular Activity
**Ribosome Profiling (Ribo-Seq)** is a powerful technique that gives us a "snapshot" of all the protein synthesis happening in a cell. Here's how it works:
1.  **Freeze:** We use chemicals to "freeze" ribosomes (the cellular machines that build proteins) right where they are on the RNA templates they are reading.
2.  **Trim:** We introduce an enzyme that chews up all the RNA that is *not* physically shielded by a ribosome.
3.  **Isolate & Sequence:** The small, protected RNA fragments that are left are called **"footprints."** We collect these footprints and sequence them.

The result is a high-resolution map that reveals the precise locations and density of ribosomes across the entire genome, telling us which genes are actively being translated.

However, this raw data is noisy. The two key signals we look for are:
- **High Footprint Density:** Translated regions have many more footprints.
- **Three-Base Periodicity:** Ribosomes move in steps of three bases (a "codon"). This creates a distinct, periodic signal in the data.

The challenge is that biological and experimental noise can obscure these signals, making it difficult to definitively identify translated regions.

## Our Approach: A "Weather Detective" for the Genome

To deconvolve the signal from the noise, we use a **Hidden Markov Model (HMM)**, a statistical tool perfect for analyzing sequential data where an observable pattern is driven by a "hidden" underlying state.

> **Analogy: Inferring Weather from Tree Rings**
>
> Imagine trying to reconstruct historical weather patterns (the hidden state; e.g., "Hot" or "Cold" year) just by looking at the size of ancient tree rings (the observable data; e.g., "Small," "Medium," or "Large"). An HMM can learn the statistical relationship between ring size and temperature, then infer the most probable sequence of past weather conditions.

In the context of `riboHMM2.0`, the model's components are as follows:
- **Hidden States:** These are the unobservable, true translational states along an mRNA transcript. Our model uses 17 distinct states, including core states like `5'UTR`, `INIT` (initiation), the three-state elongation cycle (`T1`, `T2`, `T3`), and `TERM` (termination).
- **Observable Data:** This is the sequence of ribosome footprint 5' end counts, which the model views through a sliding window at each nucleotide position.

The `riboHMM2.0` model is trained to learn the characteristic footprint patterns (emission probabilities) associated with each hidden state, as well as the probabilities of transitioning between states (transition probabilities). Using the powerful **Viterbi algorithm**, it can then analyze a new transcript and predict the most likely sequence of hidden states, thereby identifying the precise reading frame of translation.

## Key Results: What We Found

Our validation strategy was semi-supervised; while the model learns from the data in an unsupervised fashion, we used known gene annotations to "grade" its performance on a test set it had never seen before. This ensures our results are robust and generalizable.

`riboHMM2.0` demonstrates exceptional and consistent performance across multiple datasets and conditions. The model's primary strength is its high accuracy in identifying the correct translational reading frame.

| Dataset | Split Condition | Frame Accuracy | Start Accuracy (±3 nt) | Median Error |
| :--- | :--- | :--- | :--- | :--- |
| **Top 500 Expressed** | 80%/20% Train/Test | **100.0%** | 10.0% | 108.0 nt |
| **Top 500 Expressed** | Full (No Split) | 98.2% | 8.0% | 97.5 nt |
| **Top 1000 Expressed** | 80%/20% Train/Test | 97.0% | 5.5% | 115.5 nt |
| **Top 1000 Expressed**| Full (No Split) | 97.3% | 6.9% | 117.0 nt |

While pinpointing the exact start codon position remains a challenge (a known difficulty in the field), the near-perfect frame accuracy on test data confirms the model's robustness and ability to generalize.

<details>
<summary>Click to view detailed performance metrics</summary>

**Top 500 (80/20 Split)**
* Frame Accuracy: 100.00%
* Accuracy within ±3 nt: 10.00%
* Percentiles: 5th: -66.3 nt, 25th: 33.0 nt, Median: 108.0 nt, 75th: 207.0 nt, 95th: 730.9 nt

**Top 500 (No Split)**
* Frame Accuracy: 98.20%
* Accuracy within ±3 nt: 8.00%
* Percentiles: 5th: -72.3 nt, 25th: 15.0 nt, Median: 97.5 nt, 75th: 222.0 nt, 95th: 864.9 nt

**Top 1000 (80/20 Split)**
* Frame Accuracy: 97.00%
* Accuracy within ±3 nt: 5.50%
* Percentiles: 5th: -87.6 nt, 25th: 25.5 nt, Median: 115.5 nt, 75th: 273.0 nt, 95th: 757.2 nt

**Top 1000 (No Split)**
* Frame Accuracy: 97.30%
* Accuracy within ±3 nt: 6.90%
* Percentiles: 5th: -72.0 nt, 25th: 21.0 nt, Median: 117.0 nt, 75th: 297.0 nt, 95th: 846.9 nt
</details>

## The `riboHMM2.0` Pipeline: How It Works

The entire project is orchestrated by a single bash script (`run_pipeline.bash`), which executes a sequence of Python scripts.

1.  **`preprocess.py`: From Raw Data to Clean Signal**
    - Takes raw Ribo-Seq data (BAM files).
    - Identifies the dominant footprint length to reduce noise.
    - Aggregates footprint counts for the top N most expressed transcripts, creating a high-quality dataset.
    - Saves the processed data as an `.npz` file.

2.  **`split_data.py`: Verifying Generalization on Unseen Data**
    - Splits the transcript IDs into a training set (e.g., 80%) and a test set (e.g., 20%).
    - By training on one subset and evaluating on another, we can robustly assess the model's ability to generalize to new, unseen data.

3.  **`train.py`: Teaching the Model**
    - Initializes the HMM with a biologically-informed structure.
    - Trains the model on the training set using the Baum-Welch algorithm. The model learns the transition and emission probabilities that best explain the data.
    - Saves the trained model as a `.joblib` file.

4.  **`validate.py`: Grading the Model's Performance**
    - Loads the trained model and evaluates it against the unseen test set.
    - Compares the model's predictions to known gene annotations.
    - Generates a comprehensive PDF report (`validation_report_...`) with performance metrics and visualizations of the learned parameters and state predictions.

## Usage Guide (For Technical Users)

### Prerequisites
- Python 3.8+
- We recommend using a [Conda](https://docs.conda.io/en/latest/miniconda.html) environment.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/autokaiai/riboHMM_multinomial.git
    cd riboHMM_simple_prod
    ```

2.  **Set up the environment and install dependencies:**
    ```bash
    conda create --name ribohmm python=3.9
    conda activate ribohmm
    pip install -r requirements.txt 
    ```

### Running the Full Pipeline
The entire workflow can be executed with the `run_pipeline.bash` script. You must provide paths to your own reference and data files.

**Example Command:**
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
- `--orf-annotations`: Path to a CSV file with known ORF start and end coordinates.
- `--bam-files`: A space-separated list of input BAM files, enclosed in quotes.
- `--top-n`: The number of top-expressed transcripts to use for training and validation.
- `--model-suffix`: A unique suffix to append to all output files, helping to organize results from different runs.

## Project Structure

- `preprocess.py`: Processes raw BAM files into transcript-level read count arrays.
- `split_data.py`: Splits the processed data into training and testing sets.
- `train.py`: Trains the HMM on the training dataset.
- `validate.py`: Evaluates the trained HMM on the test set and generates a validation report.
- `run_pipeline.bash`: A bash script that orchestrates the entire workflow.
- `model_config.json`: Configuration file specifying the HMM architecture and initial parameters.
- `utils.py`: Contains shared helper functions used across the pipeline.
- `requirements.txt`: A list of all Python dependencies.
- `01_Reference_Files/`: Directory for reference genome/transcriptome files.
- `02_Processed_Data/`: Default output directory for processed datasets.
- `04_Models/`: Default output directory for trained models.
- `05_Reports/`: Default output directory for validation reports.

## Conclusion & Future Work

`riboHMM2.0` is a successful proof-of-concept that demonstrates the power of HMMs for interpreting Ribo-Seq data. By focusing on a robust and modern implementation, we have created a valuable tool for the genomics community.

**Strengths:** High accuracy in reading frame identification; robust to experimental noise due to its unique "lagging window" design; easy-to-use and modular pipeline.

**Limitations:** Lower precision in exact start-site identification; currently simplifies the data by only using the single most common footprint length.

**Future Directions:**
- Incorporate multiple read lengths into a unified model.
- Experiment with more complex emission models (e.g., `GaussianMixtureHMM`).
- Integrate other biological signals (like sequence context) to improve start site prediction.

## How to Cite

If you use this software for your research, please cite it as follows:

**Plain Text:**
> Kai Wöllstein, Marcel Schulz, and Christina Kalk. riboHMM_multinomial. Version 1.0.0. 2025. GitHub. https://github.com/autokaiai/riboHMM_multinomial.

**BibTeX:**
```bibtex
@software{Woellstein_2025_riboHMM,
  author       = {Kai W{\"o}llstein and Marcel Schulz and Christina Kalk},
  title        = {riboHMM\_multinomial},
  year         = {2025},
  publisher    = {GitHub},
  version      = {1.0.0},
  url          = {https://github.com/autokaiai/riboHMM_multinomial},
  urldate      = {2025-08-03}
}
```

## License

Copyright 2025 Kai Wöllstein, Marcel Schulz, Christina Kalk

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
