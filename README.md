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
https://drive.google.com/file/d/1E-k1yPH3zLvZUw0qkRGP0spjKxoP_U8O/view?usp=sharing

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
`1_Preprocessing/dataset_processors/processor_ml-1m.ipynb` loads `CausalI2I_artifacts/Datasets/Raw/ml-1m/ratings.dat`, builds train/test splits, reindexes users/items, and writes `CausalI2I_artifacts/Datasets/Processed/ml-1m/train.csv`, `test.csv`, `data_sasrec.csv`, plus `item_dict.pkl` and `Chosen_Pairs/ml-1m_chosen_pairs.pkl`.  
`1_Preprocessing/dataset_processors/processor_steam.ipynb` does the same for `CausalI2I_artifacts/Datasets/Raw/steam/steam_filtered.csv` and writes the processed Steam files.  
`1_Preprocessing/dataset_processors/processor_goodreads.ipynb` does the same for `CausalI2I_artifacts/Datasets/Raw/goodreads/goodreads_filtered.csv`, with extra duplicate handling, and writes the processed Goodreads files.  
`1_Preprocessing/descriptions/describer.ipynb` reads the processed `train.csv`/`test.csv` and writes a dataset summary text file like `1_Preprocessing/descriptions/<DATASET>_description.txt`.  
Pick the dataset(s) you want and run the matching processor notebook(s). These outputs are required by every later step.

2. **2_Propensities**  
`2_Propensities/train_MF.ipynb` trains a Matrix Factorization model on `CausalI2I_artifacts/Datasets/Processed/<DATASET>/train.csv` (with validation on `test.csv`) and saves:
- `CausalI2I_artifacts/Propensity_Models/MF_model_<DATASET>.pt`
- `CausalI2I_artifacts/Propensity_Models/MF_params_<DATASET>.pkl`  
These files are used by the later baseline and evaluation steps.

`2_Propensities/calibration_analysis.ipynb` is an optional follow-up notebook that loads the trained MF model and checks probability calibration on the processed dataset. It writes:
- `CausalI2I_artifacts/Figures/<DATASET>/calibration_plot.jpeg`
- `CausalI2I_artifacts/Propensity_Models/Calibration_Pickle.pkl`

3. **3_ChatGPT**  
*Note: You do not have to run this step if `CausalI2I_artifacts` already contains GPT results from previous runs.* 
`3_ChatGPT/launch_GPT.py` is the recommended entry point. It asks you to choose the dataset and prompt, validates `Chosen_Pairs/<DATASET>_chosen_pairs.pkl`, and builds a `nohup` command for `run_GPT.py`.  
`3_ChatGPT/run_GPT.py` sends the chosen title pairs to the OpenAI API (model `gpt-5.2`) in batches and writes results to `CausalI2I_artifacts/API_Results/<DATASET>/causal_scores_final_YYYY-MM-DD.csv` (plus a partial file during the run).  
You must have `~/secret_api_key.txt` with your API key, and the `prompts/` files must match the dataset you select.

4. **4_Baselines**  
This stage prepares the learned baseline models used in the main evaluation.

`4_Baselines/4.1_SASRec/SASRec_train.ipynb` loads `CausalI2I_artifacts/Datasets/Processed/<DATASET>/data_sasrec.csv`, trains a SASRec model, and saves:
- `CausalI2I_artifacts/SASRec_Models/sasrec_<DATASET>.pt`
- `CausalI2I_artifacts/SASRec_Models/sasrec_<DATASET>_init_dict.pkl`

`4_Baselines/4.2_OutcomeModel/prepare_dataset.ipynb` uses the processed dataset and `Chosen_Pairs/<DATASET>_chosen_pairs.pkl` to create:
- `CausalI2I_artifacts/Datasets/Processed/<DATASET>/om_train.csv`
- `CausalI2I_artifacts/Datasets/Processed/<DATASET>/om_test.csv`

`4_Baselines/4.2_OutcomeModel/train_OM.ipynb` loads those files together with the MF propensity model and saves:
- `CausalI2I_artifacts/Outcome_Models/OM_<DATASET>.pt`

The outputs of all three notebooks are consumed by step 5.

5. **5_Evaluation**  
This stage now has four notebooks.

`5_Evaluation/5.1_pre-calcuated_metrics.ipynb` precomputes baseline scores for the chosen item pairs using the MF, SASRec, and Outcome Model checkpoints, and writes:
- `CausalI2I_artifacts/Datasets/Evaluated/SASRec/<DATASET>_sasrec_scores.pkl`
- `CausalI2I_artifacts/Datasets/Evaluated/Outcome_Model/<DATASET>_om_scores.pkl`

`5_Evaluation/5.2_calculate_metrics.ipynb` loads the processed dataset, `Chosen_Pairs/<DATASET>_chosen_pairs.pkl`, GPT results from `API_Results`, the MF model, and the cached SASRec / Outcome Model scores from `5.1`. It computes the causal metrics and writes:
- `CausalI2I_artifacts/Datasets/Evaluated/<DATASET>_evaluated.csv`

`5_Evaluation/5.3_comparison.ipynb` reads that evaluated file and generates figures in `CausalI2I_artifacts/Figures/<DATASET>/`, including:
- `ate_ste_vs_causal_effect.jpeg`
- `precision_recall_at_k.jpeg`
- `pr_roc_curves.jpeg`
- `metric_distribution_by_causal_effect.jpeg`
- `ate_ste_performance_by_ess_bin.jpeg`

`5_Evaluation/5.4_make_tables.ipynb` reads the evaluated CSV files and prints the LaTeX summary table used in the paper.

6. **6_Sequels**  
This stage evaluates the sequel-specific Goodreads subset.

`6_Sequels/6.1_train_test_split.ipynb` builds a sequel-only dataset from Goodreads using `CausalI2I_artifacts/Datasets/Sequels/name2series.pkl` together with `Datasets/Processed/goodreads/*`, then writes:
- `CausalI2I_artifacts/Datasets/Sequels/train.csv`
- `CausalI2I_artifacts/Datasets/Sequels/test.csv`
- `CausalI2I_artifacts/Datasets/Sequels/id2info.pkl`

`6_Sequels/6.2_train_MF.ipynb` trains an MF model on the sequels dataset and saves:
- `CausalI2I_artifacts/Propensity_Models/MF_sequels.pt`

`6_Sequels/6.3_OutcomeModel/prepare_dataset.ipynb` creates:
- `CausalI2I_artifacts/Datasets/Sequels/om_train.csv`
- `CausalI2I_artifacts/Datasets/Sequels/om_test.csv`

`6_Sequels/6.3_OutcomeModel/train_OM.ipynb` trains the sequel Outcome Model and saves:
- `CausalI2I_artifacts/Outcome_Models/OM_sequels.pt`

`6_Sequels/6.4_calculate_metrics.ipynb` evaluates sequel metrics using `MF_sequels.pt`, `OM_sequels.pt`, and the Goodreads SASRec model (`sasrec_goodreads.pt`), then writes:
- `CausalI2I_artifacts/Datasets/Sequels/sequels_evaluated.csv`

`6_Sequels/6.5_comparison.ipynb` reads that file and generates:
- `CausalI2I_artifacts/Figures/sequels/Binned Precision.jpeg`

7. **7_Simulation**  
This stage reproduces the synthetic-data experiment.

`7_Simulation/7.1_train_test_split.ipynb` loads `CausalI2I_artifacts/Datasets/Simulation/synthetic.csv` and `ground_truth.csv`, builds train/test splits, and writes:
- `CausalI2I_artifacts/Datasets/Simulation/train.csv`
- `CausalI2I_artifacts/Datasets/Simulation/test.csv`
- `CausalI2I_artifacts/Datasets/Simulation/data_sasrec.csv`
- `CausalI2I_artifacts/Datasets/Simulation/ground_truth_processed.csv`

`7_Simulation/7.2_train_MF.ipynb` trains an MF model on the simulation data and saves:
- `CausalI2I_artifacts/Propensity_Models/MF_simulation.pt`

`7_Simulation/7.3_Baselines/SASRec_train.ipynb` trains the simulation SASRec model and saves:
- `CausalI2I_artifacts/SASRec_Models/sasrec_simulation.pt`
- `CausalI2I_artifacts/SASRec_Models/sasrec_simulation_init_dict.pkl`

`7_Simulation/7.3_Baselines/prepare_dataset.ipynb` creates:
- `CausalI2I_artifacts/Datasets/Simulation/om_train.csv`
- `CausalI2I_artifacts/Datasets/Simulation/om_test.csv`

`7_Simulation/7.3_Baselines/train_OM.ipynb` trains the simulation Outcome Model and saves:
- `CausalI2I_artifacts/Outcome_Models/OM_simulation.pt`

`7_Simulation/7.4_calculate_metrics.ipynb` evaluates causal metrics against the processed ground truth using the MF, SASRec, and Outcome Model checkpoints, then writes:
- `CausalI2I_artifacts/Datasets/Simulation/simulation_evaluated.csv`

`7_Simulation/7.5_comparison.ipynb` creates simulation figures in `CausalI2I_artifacts/Figures/simulation/`, including:
- `precision_recall_synthetic.jpeg`
- `pr_roc_synthetic.jpeg`
- `distribution_bins_synthetic.jpeg`
- `scatter_ate_vs_ste_causal_effect_(linked_only).jpeg`
