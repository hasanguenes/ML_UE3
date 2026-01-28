# General Information

This file aims to describe the structure / organisation of the project folder.

First, we want to mention that we were inspired by external resources and AI during the implmenetation procedure. The external resources are linked in comments when relevant. 


# Project Structure

```text
ML_UE3/
├── data/
│   ├── cifar-10-batches-py/      # CIFAR-10 dataset in its original batch format
│   ├── GTSRB/                    # German Traffic Sign Recognition Benchmark (GTSRB) dataset
│   ├── cifar_loader.py           # Dataset loader for CIFAR-10
│   ├── gtsrb_loader.py           # Dataset loader for GTSRB
│   └── README                    # Describes how the Dataset loaders expect the dataset folders to be structured
├── DL_approach/
│   ├── LeNet.py                  # Implementation of the LeNet-5 convolutional neural network
│   ├── resnet.py                 # ResNet18 architecture (optionally using ImageNet pretraining)
│   └── train_utils.py            # Shared training, validation, and evaluation utilities
│
├── runs/
│   └── LENET5/                   # Examplary stored experiment runs for LeNet-5 (models, histories, configs, evaluations)
│
├── analyse.ipynb                 # Notebook for ResNet18 results analysis
├── analysis_lenet.ipynb          # Notebook for LeNet5 results analysisexperiments
├── cifar_randoms.ipynb           # Random-baseline experiments for CIFAR-10
├── gtsrb_randoms.ipynb           # Random-baseline experiments for GTSRB
│
├── hyperparameter_sweep.py       # 
├── run_experiment.py             # Central CLI entry point for training and evaluation of DL models
├── run_experiment_mac.py         # 
├── script_lenet_models.py        # Helper script for running run_experiment.py multiple times
│
│── requirements.txt              # COntains the needed libraries for running Python files
│
├── README.md                     # Project overview, setup instructions, and usage guidelines
```

# Requirements

Python version: 3.13.5

You can download the necessary libraries by the following:

```text
pip install -r requirements.txt
```

# Expected data structure

das mit download erklären

# Running the DL models

You can use `run_experiment.py` to run one of the DL models. You can set a specific configuration and train the model with this script.

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

# Information about ML algorithms