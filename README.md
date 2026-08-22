# CausalI2I

Welcome to the official repository for **CausalI2I**, containing the full codebase and experimental pipeline used in our study.  
This repository is organized to support full reproducibility of the preprocessing, modeling, GPT-based evaluation, and downstream analyses described in the paper.

---

## Paper & Auxiliary Material

This repository accompanies the paper:

**“Towards Causal Item–Item Link Mining in Organic CF Data”**

- 📎 Auxiliary material: [`Auxilary_Material_to_Main_Paper.pdf`](./Auxilary_Material_to_Main_Paper.pdf)

The auxiliary document includes:
- Additional derivations for the ATE standard deviation estimator
- Full synthetic data generation procedure and oracle ground truth
- Complete LLM labeling protocol
- Full specification of the Outcome Model (OM)
- Extended experimental results

For full methodological details beyond the main text, please refer to the auxiliary material.

---

## Required Artifacts

To keep this repository lightweight and suitable for version control, large intermediate files (processed datasets, trained models, and experimental outputs) are stored separately.

Before running the code, please download the required artifacts archive:

**CausalI2I_artifacts.zip**  
Download link:
https://drive.google.com/file/d/1_IlzkCRc1baSRf8MeyL8Z1Uoq6ySI-WQ/view?usp=drive_link
 
After downloading:

1. Unzip the archive.
2. Ensure the extracted folder is named exactly `CausalI2I_artifacts`.
3. Place it in the same parent directory as this repository (as a sibling folder).

Your directory structure should look like this:
```
Home directory
  ├── CausalI2I
  └── CausalI2I_artifacts
```
The code assumes this layout when resolving paths.

---

## Prerequisites

The project is implemented in Python and depends on standard scientific-computing and machine-learning libraries.  
Please ensure your environment includes:

- `numpy`
- `pandas`
- `torch`
- `tqdm`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `openai` (required only for the GPT-based evaluation step)

For the GPT evaluation stage, place your OpenAI API key in:
```
~/secret_api_key.txt
```

---

## Execution and Reproducibility

The provided `CausalI2I_artifacts` folder contains **all intermediate datasets, trained models, GPT outputs, and evaluation files** required to reproduce the final results reported in the paper.

This means:

- You do **not** need to rerun the full pipeline.
- You may start from **any stage** of the project.
- You can regenerate only the specific figures, tables, or components you are interested in, provided that the corresponding inputs already exist in `CausalI2I_artifacts`.

If you wish to fully regenerate the entire pipeline from raw data (preprocessing → modeling → GPT labeling → evaluation → figures), follow the complete step-by-step execution order described below.

Otherwise, you may directly execute the notebook corresponding to the component you want to reproduce (e.g., evaluation or figure generation).

### Important

- Run each notebook from its own folder.  
- The project relies on `Path.cwd()` and relative parent paths to locate `CausalI2I_artifacts`.  
- Running notebooks from other directories may cause path resolution issues.

---


1. **1_Preprocessing**  
`1_Preprocessing/preprocess.ipynb` is the unified preprocessing entry point.  
It prepares dataset-specific train/test interactions, item/user indexing artifacts, and downstream inputs used by propensity, GPT, and evaluation stages.  
Run this first for the dataset(s) you want to reproduce.

2. **2_Propensities**  
This stage trains and calibrates the **SR-based propensity model** (SASRec-style).

- `2_Propensities/2.1_train_SR.ipynb`  
  Trains the sequential propensity model and saves checkpoints/parameters.
- `2_Propensities/2.2_calibration_analysis.ipynb`  
  Evaluates and calibrates propensity outputs.
- `2_Propensities/2.3_prepare_windows.ipynb`  
  Builds the window/pair-level inputs required for GPT labeling and metric computation.
- `2_Propensities/SASRec_class.py`  
  Model definition used by the SR training/evaluation notebooks.

3. **3_ChatGPT** *(optional if cached API outputs already exist)*  
LLM-based causal labeling for item pairs/windows.

- `3_ChatGPT/launch_GPT.py`  
  Launcher/runner helper for GPT jobs.
- `3_ChatGPT/run_GPT.py`  
  Executes API calls and writes causal score outputs.
- `3_ChatGPT/prompts/`  
  Dataset-specific prompt templates:
  - `prompt_ml-10m_2026-08-06.txt`
  - `prompt_steam_2026-01-11.txt`
  - `prompt_goodreads_2026-01-19.txt`

4. **4_Baselines**  
This stage currently contains the **Matrix Factorization baseline**.

- `4_Baselines/4.1_Matrix_Factorization/train_MF.ipynb`  
  Trains MF baseline model(s) used for comparison.
- `4_Baselines/4.1_Matrix_Factorization/MF_class.py`  
  MF model implementation.

5. **5_Evaluation**  
Main metrics, plots, and tables for the core experiments.

- `5_Evaluation/5.1_calculate_metrics.ipynb`  
  Computes evaluation metrics from model outputs and labeled windows/pairs.
- `5_Evaluation/5.2_comparison.ipynb`  
  Generates comparative analyses/figures across methods.
- `5_Evaluation/5.3_make_tables.ipynb`  
  Produces final summary tables (paper-ready outputs).

6. **6_Sequels**  
Goodreads sequel-focused analysis pipeline.

- `6_Sequels/6.1_choose_series.ipynb`  
  Selects/constructs sequel series subsets.
- `6_Sequels/6.2_prepare_windows.ipynb`  
  Creates sequence windows for sequel evaluation.
- `6_Sequels/6.3_calculate_metrics.ipynb`  
  Computes sequel-specific metrics.
- `6_Sequels/6.4_comparison.ipynb`  
  Generates sequel comparison figures.

7. **7_Simulation**  
Synthetic-data validation pipeline.

- `7_Simulation/7.0_generate_data.ipynb`  
  Generates synthetic interactions and ground-truth structures.
- `7_Simulation/7.1.1_train_SR.ipynb`  
  Trains SR propensity model on simulation data.
- `7_Simulation/7.1.2_calibration_analysis.ipynb`  
  Calibration analysis for simulated SR outputs.
- `7_Simulation/7.1.3_prepare_windows.ipynb`  
  Prepares simulation windows for scoring/evaluation.
- `7_Simulation/7.2_train_MF.ipynb`  
  Trains MF baseline on simulation data.
- `7_Simulation/7.3_calculate_metrics.ipynb`  
  Computes simulation metrics.
- `7_Simulation/7.4_comparison.ipynb`  
  Produces simulation comparison plots.

---
