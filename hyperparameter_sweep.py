"""
Interactive Hyperparameter Sweep for Deep Learning Models
==========================================================
Asks for dataset and model selection at runtime

Supports:
- Models: LeNet5, ResNet18
- Datasets: GTSRB, CIFAR10
"""
import subprocess
import itertools
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys

# ============================================================================
# DATASET PATHS
# ============================================================================

DATASET_PATHS = {
    "gtsrb": "data/GTSRB",
    "cifar10": "data",  # Contains cifar-10-batches-py/
}

# ============================================================================
# MODEL-SPECIFIC PARAMETER GRIDS
# ============================================================================

# LeNet5 Hyperparameters
LENET_GRID = {
    "dropout": [0.0, 0.2, 0.5],
    "lr": [1e-3],  # , 5e-4
    "augment": [0, 1],
    "epochs": [20],
    "activation": ["tanh"],  # Can add "relu" for more experiments
    "adapt_lenet": [0],
}

# ResNet18 Hyperparameters  
RESNET_GRID = {
    "dropout": [0.0, 0.2, 0.5],
    "lr": [1e-3],  # , 5e-4
    "augment": [0, 1],
    "epochs": [20],  # Models converge after ~10 epochs, 15 ensures full convergence
    "pretrained": [0, 1],      # Always pretrained for transfer learning
    "freeze_backbone": [0, 1], # Always frozen for faster training
}
# Total: 3 × 2 × 2 × 1 = 12 experiments (~6-8 hours)

# ============================================================================
# BASE CONFIGURATIONS
# ============================================================================

def get_base_config(model: str, dataset: str) -> Dict:
    """Returns base configuration for given model and dataset"""
    
    # Common settings
    base = {
        "model": model,
        "dataset": dataset,
        "data_root": DATASET_PATHS[dataset],
        "normalize": 1,
        "num_workers": 0,
        "device": "auto", # or cuda
        "save_run": 1,
        "seed": 42,
        "optimizer": "adam",
        "weight_decay": 1e-4,
        "batch_size": 128,
        "debug_fraction": 1.0,  # Change to 0.1 for quick testing
    }
    
    # Dataset-specific settings
    if dataset == "cifar10":
        base["img_size"] = 32
    elif dataset == "gtsrb":
        base["img_size"] = 32  # Can also use 64
    
    return base

# ============================================================================
# USER INPUT
# ============================================================================

def get_user_choice():
    """Interactive prompts for dataset and model selection"""
    
    print("\n" + "="*80)
    print("INTERACTIVE HYPERPARAMETER SWEEP")
    print("="*80)
    
    # Dataset selection
    print("\nAvailable Datasets:")
    print("  1) GTSRB  - German Traffic Sign Recognition")
    print("  2) CIFAR10 - 10-class image classification")
    
    while True:
        dataset_choice = input("\nSelect dataset (1 or 2): ").strip()
        if dataset_choice == "1":
            dataset = "gtsrb"
            break
        elif dataset_choice == "2":
            dataset = "cifar10"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Model selection
    print("\nAvailable Models:")
    print("  1) LeNet5  - Classic CNN (simple, fast)")
    print("  2) ResNet18 - Deeper network (pretrained + frozen)")
    
    while True:
        model_choice = input("\nSelect model (1 or 2): ").strip()
        if model_choice == "1":
            model = "lenet5"
            break
        elif model_choice == "2":
            model = "resnet18"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    print("\n" + "="*80)
    print(f"Selected: {model.upper()} on {dataset.upper()}")
    print("="*80)
    
    return model, dataset

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def estimate_total_time(num_experiments: int, avg_epochs: float, quick_mode: bool = False) -> str:
    """Estimates total runtime"""
    if quick_mode:
        minutes_per_epoch = 0.5
    else:
        minutes_per_epoch = 2.0
    
    total_minutes = num_experiments * avg_epochs * minutes_per_epoch
    hours = total_minutes / 60
    
    if hours < 1:
        return f"~{int(total_minutes)} minutes"
    elif hours < 24:
        return f"~{hours:.1f} hours"
    else:
        days = hours / 24
        return f"~{days:.1f} days"


def calculate_grid_size(param_grid: Dict) -> int:
    """Calculates total number of experiments"""
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def build_command(config: Dict[str, Any]) -> List[str]:
    """Builds the CLI command"""
    cmd = [sys.executable, "run_experiment_mac.py", "--mode", "train"]
    
    for key, value in config.items():
        cli_key = key.replace("_", "-")
        cmd.extend([f"--{cli_key}", str(value)])
    
    return cmd


def generate_configs(base_config: Dict, param_grid: Dict) -> List[Dict]:
    """Generates all combinations, filtering out invalid combinations"""
    keys = param_grid.keys()
    values = param_grid.values()
    
    configs = []
    invalid_count = 0
    
    for combination in itertools.product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combination)))
        
        # CONSTRAINT: if pretrained=0, then freeze_backbone must be 0
        # (no sense to freeze random weights)
        if config.get("pretrained") == 0 and config.get("freeze_backbone") == 1:
            invalid_count += 1
            continue
        
        configs.append(config)
    
    if invalid_count > 0:
        print(f"\n⚠ Filtered out {invalid_count} invalid configuration(s):")
        print(f"  (pretrained=0 with freeze_backbone=1 are skipped)")
    
    return configs


def run_experiment(config: Dict, exp_id: int, total: int) -> Dict:
    """Runs a single experiment with progress tracking"""
    cmd = build_command(config)
    
    print("\n" + "="*80)
    print(f"EXPERIMENT {exp_id}/{total}")
    print("="*80)
    print(f"Model: {config['model']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Augmentation: {config['augment']}")
    print(f"Learning Rate: {config['lr']}")
    print(f"Dropout: {config['dropout']}")
    print(f"Epochs: {config['epochs']}")
    
    if config['model'] == 'lenet5':
        print(f"Activation: {config.get('activation', 'N/A')}")
    else:
        print(f"Pretrained: {config.get('pretrained', 'N/A')}")
        print(f"Frozen Backbone: {config.get('freeze_backbone', 'N/A')}")
    
    print("="*80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        # Extract run directory from output
        run_dir = None
        for line in result.stdout.split('\n'):
            if 'Saved run to:' in line:
                run_dir = line.split('Saved run to:')[-1].strip()
                break
        
        print(f"✓ SUCCESS in {elapsed/60:.1f} minutes")
        if run_dir:
            print(f"  Run dir: {run_dir}")
        
        return {
            "status": "success",
            "experiment_id": exp_id,
            "config": config,
            "run_dir": run_dir,
            "elapsed_time_minutes": elapsed / 60
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ FAILED after {elapsed/60:.1f} minutes")
        error_msg = e.stderr[-500:] if e.stderr else str(e)
        print(f"Error: {error_msg}")
        
        return {
            "status": "failed",
            "experiment_id": exp_id,
            "config": config,
            "error": error_msg,
            "elapsed_time_minutes": elapsed / 60
        }


def print_summary_statistics(results: List[Dict]) -> None:
    """Prints summary statistics"""
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        total_time = sum(r["elapsed_time_minutes"] for r in successful)
        avg_time = total_time / len(successful)
        print(f"Total training time: {total_time/60:.1f} hours")
        print(f"Average time per experiment: {avg_time:.1f} minutes")
    
    print("="*80)


def create_analysis_script(results_file: Path, model: str, dataset: str) -> None:
    """Creates a Python script to analyze results"""
    
    analysis_code = f'''
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

print("="*80)
print("ANALYSIS: {model.upper()} on {dataset.upper()}")
print("="*80)

# Load results
with open("{results_file}", "r") as f:
    results = json.load(f)

# Extract successful experiments
successful = [r for r in results if r["status"] == "success"]
print(f"\\nSuccessful experiments: {{len(successful)}}")
print(f"Failed experiments: {{len(results) - len(successful)}}")

if len(successful) == 0:
    print("\\nNo successful experiments to analyze!")
    exit(1)

# Load metrics from each run
data = []
for result in successful:
    run_dir = Path(result["run_dir"])
    
    try:
        # Load metrics
        metrics = json.loads((run_dir / "metrics.json").read_text())
        
        # Combine config and metrics
        row = result["config"].copy()
        row["overall_accuracy"] = metrics["overall_accuracy"]
        row["balanced_accuracy"] = metrics["balanced_accuracy"]
        row["run_dir"] = str(run_dir)
        row["training_time"] = result.get("elapsed_time_minutes", 0)
        data.append(row)
    except Exception as e:
        print(f"Warning: Could not load metrics from {{run_dir}}: {{e}}")

if not data:
    print("\\nNo valid metrics found!")
    exit(1)

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("experiment_results_detailed.csv", index=False)
print(f"\\nResults saved to: experiment_results_detailed.csv")

# ============================================================================
# ANALYSIS: Augmentation Impact (MAIN QUESTION)
# ============================================================================
print("\\n" + "="*80)
print("PRIMARY ANALYSIS: AUGMENTATION IMPACT")
print("="*80)

aug_comparison = df.groupby("augment").agg({{
    "overall_accuracy": ["mean", "std", "min", "max", "count"],
    "balanced_accuracy": ["mean", "std"],
}}).round(4)

print("\\n", aug_comparison)

# Statistical test
from scipy import stats
no_aug = df[df["augment"] == 0]["overall_accuracy"]
with_aug = df[df["augment"] == 1]["overall_accuracy"]

if len(no_aug) > 0 and len(with_aug) > 0:
    t_stat, p_value = stats.ttest_ind(no_aug, with_aug)
    improvement = (with_aug.mean() - no_aug.mean()) * 100
    
    print(f"\\nStatistical Test:")
    print(f"  Without Aug: {{no_aug.mean():.4f}} +/- {{no_aug.std():.4f}}")
    print(f"  With Aug:    {{with_aug.mean():.4f}} +/- {{with_aug.std():.4f}}")
    print(f"  Improvement: {{improvement:+.2f}}%")
    print(f"  P-value:     {{p_value:.6f}}")
    
    if p_value < 0.001:
        print("  Result: HIGHLY SIGNIFICANT (p < 0.001) ***")
    elif p_value < 0.01:
        print("  Result: VERY SIGNIFICANT (p < 0.01) **")
    elif p_value < 0.05:
        print("  Result: SIGNIFICANT (p < 0.05) *")
    else:
        print("  Result: NOT significant (p >= 0.05)")

# ============================================================================
# TOP CONFIGURATIONS
# ============================================================================
print("\\n" + "="*80)
print("TOP 10 CONFIGURATIONS")
print("="*80)

top10 = df.nlargest(10, "overall_accuracy")
for col in ["augment", "dropout", "lr", "epochs", "overall_accuracy", "training_time"]:
    if col in top10.columns:
        print(f"\\n{{col}}:")
        print(top10[col].tolist())

# ============================================================================
# VISUALIZATIONS
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"{model.upper()} on {dataset.upper()} - Hyperparameter Analysis", fontsize=16)

# 1. Augmentation Impact
ax = axes[0, 0]
df.boxplot(column="overall_accuracy", by="augment", ax=ax)
ax.set_title("Augmentation Impact", fontsize=12, fontweight='bold')
ax.set_xlabel("Augmentation")
ax.set_ylabel("Overall Accuracy")
ax.get_figure().suptitle("")
ax.set_xticklabels(['No', 'Yes'])

# 2. Dropout Impact
ax = axes[0, 1]
dropout_pivot = df.pivot_table(values="overall_accuracy", index="dropout", columns="augment", aggfunc="mean")
dropout_pivot.plot(kind="bar", ax=ax, rot=0)
ax.set_title("Dropout Impact")
ax.legend(["No Aug", "With Aug"])
ax.grid(axis='y', alpha=0.3)

# 3. Learning Rate Impact
ax = axes[0, 2]
lr_pivot = df.pivot_table(values="overall_accuracy", index="lr", columns="augment", aggfunc="mean")
lr_pivot.plot(kind="bar", ax=ax, rot=45)
ax.set_title("Learning Rate Impact")
ax.legend(["No Aug", "With Aug"])
ax.grid(axis='y', alpha=0.3)

# 4. Epochs Learning Curve
ax = axes[1, 0]
for aug in [0, 1]:
    subset = df[df["augment"] == aug].groupby("epochs")["overall_accuracy"].mean()
    ax.plot(subset.index, subset.values, marker='o', label=f"{{'With' if aug else 'Without'}} Aug", linewidth=2)
ax.set_title("Epochs vs Accuracy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Accuracy Distribution
ax = axes[1, 1]
for aug_val in [0, 1]:
    subset = df[df["augment"] == aug_val]["overall_accuracy"]
    ax.hist(subset, alpha=0.6, bins=15, label=f"Aug={{aug_val}}", edgecolor='black')
ax.set_title("Accuracy Distribution")
ax.set_xlabel("Overall Accuracy")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 6. Training Time vs Accuracy
ax = axes[1, 2]
for aug in [0, 1]:
    subset = df[df["augment"] == aug]
    ax.scatter(subset["training_time"], subset["overall_accuracy"], 
               alpha=0.6, s=50, label=f"Aug={{aug}}")
ax.set_title("Training Efficiency")
ax.set_xlabel("Training Time (min)")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hyperparameter_analysis.png", dpi=300, bbox_inches="tight")
print("\\nPlots saved to: hyperparameter_analysis.png")

plt.show()
'''
    
    analysis_path = Path("analyze_results.py")
    analysis_path.write_text(analysis_code, encoding='utf-8')
    print(f"\n✓ Created analysis script: {analysis_path}")
    print("  Run it after experiments: python analyze_results.py")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Get user choices
    model, dataset = get_user_choice()
    
    # Get appropriate config and grid
    base_config = get_base_config(model, dataset)
    param_grid = LENET_GRID if model == "lenet5" else RESNET_GRID
    
    # Calculate experiment info
    num_experiments = calculate_grid_size(param_grid)
    avg_epochs = sum(param_grid["epochs"]) / len(param_grid["epochs"])
    quick_mode = base_config["debug_fraction"] < 1.0
    
    # Print experiment info
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Model: {model.upper()}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Data path: {base_config['data_root']}")
    print(f"\nParameter Grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print(f"\nFixed Parameters:")
    print(f"  Optimizer: {base_config['optimizer']}")
    print(f"  Weight Decay: {base_config['weight_decay']}")
    print(f"  Batch Size: {base_config['batch_size']}")
    print(f"\nTotal experiments: {num_experiments}")
    print(f"Debug mode: {quick_mode} (fraction={base_config['debug_fraction']})")
    print(f"Estimated time: {estimate_total_time(num_experiments, avg_epochs, quick_mode)}")
    print("="*80)
    
    # Confirmation
    if not quick_mode:
        response = input(f"\nThis will run {num_experiments} experiments. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Generate configs
    configs = generate_configs(base_config, param_grid)
    print(f"\nGenerated {len(configs)} configurations. Starting experiments...\n")
    
    # Run experiments
    results = []
    start_time = time.time()
    
    for i, config in enumerate(configs, 1):
        result = run_experiment(config, i, len(configs))
        results.append(result)
        
        # Progress update
        elapsed_total = (time.time() - start_time) / 60
        if i > 1:
            avg_time = elapsed_total / i
            remaining = (len(configs) - i) * avg_time
            print(f"\nProgress: {i}/{len(configs)} ({i/len(configs)*100:.1f}%)")
            print(f"Elapsed: {elapsed_total:.1f} min | Est. remaining: {remaining:.1f} min")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"results_{model}_{dataset}_{timestamp}.json")
    results_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
    
    # Print summary
    print_summary_statistics(results)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Create analysis script
    create_analysis_script(results_file, model, dataset)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print(f"1. Review results: {results_file}")
    print("2. Run analysis: python analyze_results.py")
    print("3. Check plots: hyperparameter_analysis.png")
    print("4. Check CSV: experiment_results_detailed.csv")
    print(f"5. Individual runs: runs/{model.upper()}/{dataset.upper()}/")
    print("="*80)


if __name__ == "__main__":
    main()