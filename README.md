# General Information

This file aims to describe the structure / organisation of the project folder.

First, we want to mention that we were inspired by external resources and AI during the implmenetation procedure. The external resources are linked in comments when relevant. 


# Project Structure

```text
ML_UE3/
├── data/
│   ├── cifar_loader.py           # Dataset loader for CIFAR-10
│   ├── gtsrb_loader.py           # Dataset loader for GTSRB
│   └── README                    # Describes expected dataset folder structure
│
├── DL_approach/
│   ├── LeNet.py                  # Implementation of LeNet-5 CNN
│   ├── resnet.py                 # ResNet18 architecture (supports ImageNet pretraining)
│   └── train_utils.py            # Shared training, validation, and evaluation utilities
│
├── runs/                         # Stores experiment results (Deep Learning)
├── runs_ml/                      # Stores experiment results (Shallow Learning)
│
├── analysis_lenet.ipynb          # Notebook for LeNet5 results analysis
├── analysis_resnet.ipynb         # Notebook for ResNet18 results analysis
├── cifar_randoms.ipynb           # Random-baseline experiments for CIFAR-10
├── gtsrb_randoms.ipynb           # Random-baseline experiments for GTSRB
│
├── feature_extractions.py        # Feature extraction modules (SIFT + BoVW, 3D Color Histogram) for ML models
├── run_experiment.py             # CLI entry point for training DL models (LeNet/ResNet)
├── run_experiment_mac.py         # Mac-optimized CLI entry point (MPS support)
├── run_shallow_baselines.py      # Pipeline for shallow learning (Feat. Extraction -> LogisticReg/RandomForest -> Eval)
├── resnet_hyperparameter_sweep.py# Automation script to run multiple run_experiment_mac.py (works also on windows) 
├── script_lenet_models.py        # Script to run run_experiment script multiple times 
├── sift_bovw.py                  # SIFT+BOVW
├── feature_extractions.py        # Feature Extraction 
│
├── requirements.txt              # Required Python libraries
└── README.md                     # Project documentation
```

# Requirements

Python version: 3.13.5 (Windows)
Python verison: 3.12.9 (Mac)

You can download the necessary libraries by the following:

```text
pip install -r requirements.txt
```

# Expected data structure

Since the datasets are too big of a size, we provide a link where you can download the datasets already in the right structure. You just have to download this zip file:

https://1drv.ms/u/c/55f6a2f9c30abbe2/IQAtnNpT3olDR46UVxJVCASeAXLBjZmauuej0RQ22BV2oBY?e=ubMOl9

And unzip this zip file in the data folder.

# Running the DL models

You can use `run_experiment.py` to run one of the DL models. You can set a specific configuration and train the model with this script. `run_experiment_mac.py` is specifically for mac.

To see all available command-line options and their default values:

```text
python run_experiment.py --help
```

Here the options with their meanings and domains:

### Model & Execution
- `--model` : `{lenet5,resnet18}`  
  Which model to run. (default: `lenet5`)

- `--mode` : `{train,eval}`  
  Execution mode. `train` trains a new model; `eval` loads a saved run from `--run-dir` and evaluates it. (default: `train`)

---

### Dataset & Input
- `--dataset` : `{gtsrb,cifar10}`  
  Which dataset to use. (default: `gtsrb`)

- `--data-root` : `PATH`  
  Root directory of the dataset on disk. (default: `../data/GTSRB`)

- `--img-size` : `{32,64}`  
  Input image size. CIFAR-10 is forced to 32; GTSRB supports 32 or 64. (default: `32`)

- `--normalize` : `{0,1}`  
  Whether to normalize inputs using dataset mean/std. (default: `1`)

- `--augment` : `{0,1}`  
  Enable data augmentation for the training set only. (default: `0`)

- `--debug-fraction` : `FLOAT`  
  Load only this fraction of the dataset for quick debugging (e.g. `0.05`). (default: `1.0`)

---

### Training Hyperparameters
- `--epochs` : `INT`  
  Number of training epochs. (default: `10`)

- `--batch-size` : `INT`  
  Mini-batch size used during training. (default: `128`)

- `--optimizer` : `{adam,sgd}`  
  Optimizer used for training. (default: `adam`)

- `--lr` : `FLOAT`  
  Learning rate. (default: `0.001`)

- `--weight-decay` : `FLOAT`  
  L2 weight decay (regularization). (default: `0.0`)

- `--dropout` : `FLOAT`  
  Dropout probability applied in fully connected layers. Set `0.0` to disable dropout. (default: `0.0`)

---

### Architecture-Specific Options
- `--activation` : `{tanh,relu}`  
  Activation function used inside LeNet-5. (default: `tanh`)

- `--adapt-lenet` : `{0,1}`  
  Enable adaptive pooling for non-32x32 inputs. (default: `0`)

- `--pretrained` : `{0,1}`  
  ResNet18 only: load ImageNet pretrained weights. (default: `0`)

- `--freeze-backbone` : `{0,1}`  
  ResNet18 only: freeze backbone and train only the final layer. (default: `0`)

---

### Runtime & Reproducibility
- `--device` : `{auto,cpu,cuda}`  
  Compute device selection. (default: `auto`)

- `--num-workers` : `INT`  
  Number of DataLoader worker processes. (default: `0`)

- `--seed` : `INT`  
  Random seed for reproducibility. (default: `42`)

---

### Output & Storage
- `--runs-dir` : `PATH`  
  Base output directory where run folders are stored. (default: `runs`)

- `--run-dir` : `PATH`  
  Path to an existing run folder (required when using `--mode eval`). (default: empty)

- `--save-run` : `{0,1}`  
  Save run (weights, config, history, plots). (default: `1`)

---

Here all possible options:

```text
run_experiment.py [-h] [--model {lenet5,resnet18}] [--mode {train,eval}] [--dataset {gtsrb,cifar10}] [--data-root DATA_ROOT] [--runs-dir RUNS_DIR]
                         [--run-dir RUN_DIR] [--img-size IMG_SIZE] [--normalize {0,1}] [--augment {0,1}] [--debug-fraction DEBUG_FRACTION]
                         [--activation {tanh,relu}] [--dropout DROPOUT] [--adapt-lenet {0,1}] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                         [--optimizer {adam,sgd}] [--lr LR] [--weight-decay WEIGHT_DECAY] [--pretrained {0,1}] [--freeze-backbone {0,1}]
                         [--device {auto,cpu,cuda}] [--num-workers NUM_WORKERS] [--seed SEED] [--save-run {0,1}]
```

Here an example usage case:

Train LeNet-5 on the CIFAR-10 dataset without data augmentation for a single epoch:

```bash
python run_experiment.py \
  --model lenet5 \
  --dataset cifar10 \
  --epochs 1 \
  --augment 0 \
  --runs-dir runs/lenet5_cifar10
```

Train ResNet-18 on the GTSRB dataset without data augmentation for a single epoch:

```bash
python run_experiment.py \
  --model lenet5 \
  --dataset gtsrb \
  --epochs 1 \
  --augment 0 \
  --runs-dir runs/lenet5_cifar10
```

# Running the ML Models

You can use `run_shallow_baselines.py` to train shallow learning models (Logistic Regression and Random Forest) using traditional feature extraction methods (Color Histogram and SIFT + Bag of Visual Words).

The script supports both **interactive mode** (with prompts) and **CLI mode** (with command-line arguments).

To see all available command-line options and their default values:
```bash
python run_shallow_baselines.py --help
```

## Command-Line Options

### Dataset & Model Selection
- `--dataset, -d` : `{cifar10, gtsrb}`  
  Which dataset to use. If not provided, the script enters interactive mode.

- `--model, -m` : `{lr, rf, both}`  
  Which model(s) to train:
  - `lr`: Logistic Regression only
  - `rf`: Random Forest only
  - `both`: Train both models
  
- `--features, -f` : `{color, sift, both}`  
  Which feature extraction method(s) to use:
  - `color`: 3D Color Histogram
  - `sift`: SIFT + Bag of Visual Words (BoVW)
  - `both`: Extract both feature types

---

### Sample Size (Optional)
- `--train-samples` : `INT`  
  Number of training samples to use. If not specified, uses the full training set. Useful for quick debugging (e.g., `500`).

- `--test-samples` : `INT`  
  Number of test samples to use. If not specified, uses the full test set. Useful for quick debugging (e.g., `100`).

---

### Feature Extraction Parameters
- `--bins` : `INT`  
  Number of bins per channel for the 3D Color Histogram. (default: `8`)

- `--clusters` : `INT`  
  Number of visual word clusters for SIFT + BoVW. (default: `100`)

---

### Cross-Validation Settings
- `--cv-folds` : `INT`  
  Number of cross-validation folds for hyperparameter tuning. (default: `3`)

---

### Execution Modes
- `--yes, -y`  
  Skip the confirmation prompt before training starts. Useful for automation.

- `--batch`  
  Batch mode: Requires `--dataset`, `--model`, and `--features` to be specified. No interactive prompts will appear. Automatically implies `--yes`.

---

## Interactive Mode

If you run the script without arguments, it will guide you through the configuration:
```bash
python run_shallow_baselines.py
```

You'll be prompted to select:
1. **Dataset**: CIFAR-10 or GTSRB
2. **Model**: Logistic Regression, Random Forest, or both
3. **Features**: Color Histogram, SIFT+BoVW, or both
4. **Sample sizes**: Option to use a subset for quick testing
5. **Confirmation**: Review configuration before starting

---

## Example Usage

### Example 1: Full Training with All Options
Train both models on CIFAR-10 using both feature types:
```bash
python run_shallow_baselines.py -d cifar10 -m both -f both -y
```

### Example 2: Quick Debug Run
Train Logistic Regression on GTSRB with Color Histogram using only 500 train and 100 test samples:
```bash
python run_shallow_baselines.py -d gtsrb -m lr -f color --train-samples 500 --test-samples 100 -y
```

