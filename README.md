# Uncertainty quantification in LLMs

This repository contains an implementation that builds upon the work of [Semantically Diverse Language Generation for Uncertainty Estimation in Language Models (SDLG)](https://github.com/ml-jku/SDLG). This project is currently **incomplete and under active development**.

The original SDLG framework introduces a method to estimate the **aleatoric uncertainty** of a language model by generating a set of semantically diverse outputs for a given prompt. This project extends that pipeline with two key contributions:

1.  **Epistemic Uncertainty Estimation:** We integrate the estimation of epistemic uncertainty into the same generation pipeline, allowing for a more comprehensive view of the model's total uncertainty.
2.  **Improved Importance Sampling:** We explore and implement an improved importance sampling approach to better weight the generated sequences, aiming for a more accurate uncertainty estimation.

## File Descriptions

*   `run_experiments.py`: The main script for executing the generation pipeline. It loads the specified LLM, processes a dataset, and uses various methods (baseline, SDLG, DOLA-SDLG) to generate and save output sequences and their likelihoods.
*   `analyze_results.py`: Handles the post-processing and analysis of the generated results. This script computes semantic entropy, AUROC scores, and other metrics to evaluate both the aleatoric and epistemic uncertainty estimations.
*   `sdlg.py`: Contains the core implementation of the Semantically Diverse Language Generation (SDLG) algorithm, which steers the LLM to produce outputs that are both likely and semantically varied.
*   `seq_lvl_imp.py`: Implements the improved sequence-level importance calculations, a key contribution of this work for refining the uncertainty estimation.
*   `args.py`: Defines the configuration parameters and arguments for experiments, such as model identifiers, generation settings, and file paths.
*   `utils.py`: A collection of utility functions used across the project, including helpers for computing correctness, likelihoods, and semantic clustering.
*   `disagree.py`: Contains functionalities for measuring the disagreement or diversity among generated outputs, a critical component for assessing semantic uncertainty.

## Setup and Installation

### 1. Dependencies

It is recommended to use a virtual environment. Install the required Python libraries using the following commands:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade datasets
pip install evaluate
pip install git+https://github.com/google-research/bleurt.git
pip install rouge_score
pip install --upgrade sentence_transformers
pip install protobuf==3.20.*
pip install info-nce-pytorch
pip install --upgrade transformers
pip install --upgrade accelerate
pip install --upgrade bitsandbytes
```

### 2. Hugging Face Authentication

Access to certain models (e.g., Llama 2) requires authentication with the Hugging Face Hub.

1.  Obtain an API token from your Hugging Face account settings.
2.  Set this token as an environment variable. For the current terminal session, you can run:

    ```bash
    export HF_AUTH_TOKEN='your_api_token'
    ```

    For a more permanent setup, add this line to your shell's configuration file (e.g., `~/.bashrc` or `~/.zshrc`) and restart your terminal or source the file.

## How to Run

The workflow consists of two main stages: first generating results from the models, and then analyzing those results.

### 1. Run Experiments

Execute the `run_experiments.py` script to begin the generation process. You can modify experiment parameters directly in the `args.py` file.

```bash
python run_experiments.py
```

This script will save the detailed generation outputs and likelihoods as pickle (`.pkl`) files in the specified results directory.

### 2. Analyze Results

Once the generation is complete, run the analysis script to process the saved data:

```bash
python analyze_results.py
```

This will compute the final uncertainty metrics (including aleatoric and epistemic measures) and save the analysis, typically as CSV files containing AUROC scores and other evaluation data.
