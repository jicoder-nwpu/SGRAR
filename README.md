# Strategy-Grounded Reasoning and Adaptive Retrieval for Emotionally Supportive Dialogue Systems

Data and code repository for **“Strategy-Grounded Reasoning and Adaptive Retrieval for Emotionally Supportive Dialogue Systems.”**

This repository organizes the datasets and scripts used for empathetic dialogue experiments, including data preparation, supervised fine-tuning (SFT), retrieval-augmented generation (RAG), reward computation, reinforcement learning with PPO, and evaluation on the ED and ESC datasets.

## Directory Structure

```text
.
├── data/                                    # Dataset directory
│   ├── ED_json/                             # ED dataset
│   │   ├── test_data.json                   # ED test set
│   │   ├── train_data.json                  # ED training set
│   │   └── valid_data.json                  # ED validation set
│   └── ESC_json/                            # ESC dataset
│       ├── test.json                        # ESC test set
│       └── train.json                       # ESC training set
├── process/                                 # Training and processing scripts
│   ├── evaluate/                            # Evaluation-related scripts and data
│   │   ├── ED.jsonl                         # ED evaluation data
│   │   ├── ESC.jsonl                        # ESC evaluation data
│   │   ├── evaluate_ED.py                   # ED evaluation script
│   │   ├── evaluate_ESC.py                  # ESC evaluation script
│   │   └── parquet2jsonl.py                # Convert Parquet to JSONL
│   ├── ppo/                                 # PPO reinforcement learning
│   │   ├── genarate_ppo.sh                  # Generation script for PPO experiments
│   │   └── ppo.sh                           # PPO training script
│   ├── reward/                              # Reward computation
│   │   └── reward_score.py                  # Reward function for reinforcement learning
│   ├── search/                              # Retrieval and RAG operations
│   │   ├── build-ed-databse.py              # Build ED retrieval database
│   │   ├── build_ed_faiss.py                # Build ED FAISS index
│   │   ├── build-esc-databse.py             # Build ESC retrieval database
│   │   ├── build_esc_faiss.py               # Build ESC FAISS index
│   │   ├── calculate_rag_score.py           # Calculate RAG-related scores
│   │   ├── ED_test_label_rerank_rag.py      # based reranking for ED ESC data
│   │   └── ED_test_random_rerank_rag.py     # based reranking for ED ESC data
│   └── sft/                                 # Supervised fine-tuning
│       ├── genarate.sh                      # Model response generation script
│       ├── jsonl_to_parquet.py              # Convert JSONL to Parquet
│       └── sft_train.sh                     # SFT training script
└──requirments.txt                           # Dependency Management File  
```

## Main Files Introduction

### Datasets (`data/`)

- **`ED_json/`**: Contains the ED training, validation, and test datasets.
- **`ESC_json/`**: Contains the ESC training and test datasets.

### Evaluation (`process/evaluate/`)

- **`evaluate_ED.py`**: Evaluates model-generated responses on the ED dataset.
- **`evaluate_ESC.py`**: Evaluates model-generated responses on the ESC dataset.
- **`parquet2jsonl.py`**: Converts Parquet files into JSONL files for subsequent processing or evaluation.
- **`ED.jsonl` / `ESC.jsonl`**: JSONL data used in the corresponding evaluation workflow.

### Supervised Fine-Tuning (`process/sft/`)

- **`jsonl_to_parquet.py`**: Converts JSONL training data into Parquet format.
- **`sft_train.sh`**: Runs the supervised fine-tuning workflow.
- **`genarate.sh`**: Runs model response generation for SFT experiments.

### Retrieval and RAG (`process/search/`)

- **`build-ed-databse.py` / `build-esc-databse.py`**: Build retrieval databases for ED and ESC, respectively.
- **`build_ed_faiss.py` / `build_esc_faiss.py`**: Build FAISS indices for the two datasets.
- **`calculate_rag_score.py`**: Calculates retrieval-related scores used in the RAG workflow.
- **`ED_test_label_rerank_rag.py`**: Performs label-based retrieval reranking on ED test data.
- **`ED_test_random_rerank_rag.py`**: Performs random-based retrieval reranking on ED test data.

### Reward Function (`process/reward/`)

- **`reward_score.py`**: Defines the reward computation used during reinforcement learning.

### PPO Reinforcement Learning (`process/ppo/`)

- **`ppo.sh`**: Runs the PPO training workflow.
- **`genarate_ppo.sh`**: Generates model responses for PPO-related experiments.

## Execution Workflow

A typical workflow is:

1. **Prepare the data** using the files in `data/` and the format conversion scripts.
2. **Build retrieval resources (if needed)** using the scripts in `process/search/`.
3. **Run SFT** using `process/sft/sft_train.sh`.
4. **Run PPO training (if needed)** using `process/ppo/ppo.sh`, with the reward logic in `process/reward/reward_score.py`.
5. **Generate model responses** using the generation script associated with the training stage.
6. **Evaluate the generated responses** with the ED or ESC evaluation script.

### Example Commands

Run the following commands from the repository root after configuring the dataset paths, model paths, environment, and other options inside the scripts.

**SFT training**

```bash
bash process/sft/sft_train.sh
```

**SFT response generation**

```bash
bash process/sft/genarate.sh
```

**PPO training**

```bash
bash process/ppo/ppo.sh
```

**PPO response generation**

```bash
bash process/ppo/genarate_ppo.sh
```

**Evaluation and file conversion**

```bash
python process/evaluate/evaluate_ED.py --help
python process/evaluate/evaluate_ESC.py --help
python process/evaluate/parquet2jsonl.py --help
```

> **Note:** The Python scripts may use hard-coded configuration or require specific arguments rather than supporting `--help`. Check their source code before execution. The shell scripts may also need path and environment updates. The commands above describe entry points, not a verified end-to-end configuration.

## Evaluation

The repository includes dataset-specific evaluation entry points:

- **ED:** `process/evaluate/evaluate_ED.py`
- **ESC:** `process/evaluate/evaluate_ESC.py`

Evaluation implementations and metrics depend on the scripts and their imported modules. Check the corresponding evaluation code for the exact metrics, model resources, input schema, and output format used in each experiment.

## Notes

- Script names in this README match the repository.
- Some scripts may contain absolute paths to datasets, model checkpoints, or output locations. Update these paths for your environment before running them.
- SFT, retrieval, PPO, and evaluation are separated into folders to make each experimental stage easier to manage and reproduce.
