"""
Features:
- Color Histogram (3D)
- SIFT + Bag of Visual Words (BoVW)
Classifiers:
- Logistic Regression
- Random Forest
Usage: python run_shallow_baselines.py
Interactive prompts will guide you through dataset and model selection.
"""
import numpy as np
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
import cv2
import random       
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - MODIFY HYPERPARAMETERS HERE
# =============================================================================
# Random Forest hyperparameters to search
# Add/remove parameters as needed
RF_PARAM_GRID = {
    'n_estimators': [30, 60, 90],
    'max_depth': [5, 10, 15],  # or None
    # Uncomment to add more parameters:
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['sqrt', 'log2', None],
    # 'bootstrap': [True, False],
}
# Logistic Regression C values (only used when penalty is not None)
LR_C_VALUES = [0.1, 1, 10]

# Cross-validation settings
CV_FOLDS = 3
CV_SCORING = 'accuracy'  # or 'balanced_accuracy', 'f1_macro'

# Feature extraction settings
COLOR_HIST_BINS = [8]  # [6, 8, 10]
BOVW_N_CLUSTERS = 100  # Number of visual words

# Data settings
DATA_ROOT = "data"
MAX_TRAIN_SAMPLES = None  # Set to int for debugging, None for full dataset
MAX_TEST_SAMPLES = None

# =============================================================================
# RUNTIME TRACKER - Detailed runtime analysis
# =============================================================================
@dataclass
class RuntimeStats:
    """Stores runtime statistics for an operation."""
    train_time: float = 0.0
    test_time: float = 0.0
    total_time: float = 0.0
    train_samples: int = 0
    test_samples: int = 0
    
    @property
    def train_per_sample(self) -> float:
        return self.train_time / self.train_samples if self.train_samples > 0 else 0
    
    @property
    def test_per_sample(self) -> float:
        return self.test_time / self.test_samples if self.test_samples > 0 else 0

@dataclass 
class ModelResult:
    """Stores results for final summary."""
    model_name: str
    feature_name: str
    feat_time_train: float  # Feature extraction time (train)
    feat_time_test: float   # Feature extraction time (test)
    train_time: float       # Mean training time from CV (averaged over folds)
    test_time: float        # Prediction time on test set (single measurement)
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    cv_score: float         # Best CV score
    best_params: Dict
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)

@dataclass
class RuntimeTracker:
    """Central class for runtime analysis."""
    feature_extraction: Dict[str, RuntimeStats] = field(default_factory=dict)
    model_training: Dict[str, RuntimeStats] = field(default_factory=dict)
    final_results: List[ModelResult] = field(default_factory=list)
    
    def add_feature_time(self, name: str, train_time: float, test_time: float,
                         n_train: int, n_test: int):
        self.feature_extraction[name] = RuntimeStats(
            train_time=train_time, test_time=test_time,
            total_time=train_time + test_time,
            train_samples=n_train, test_samples=n_test
        )
    
    def add_model_time(self, name: str, train_time: float, test_time: float,
                       n_train: int, n_test: int):
        self.model_training[name] = RuntimeStats(
            train_time=train_time, test_time=test_time,
            total_time=train_time + test_time,
            train_samples=n_train, test_samples=n_test
        )
    
    def add_final_result(self, model_name: str, feature_name: str,
                         feat_time_train: float, feat_time_test: float,
                         train_time: float, test_time: float,
                         accuracy: float, balanced_accuracy: float, f1_macro: float,
                         cv_score: float, best_params: Dict,
                         per_class_accuracy: Dict[str, float]):
        self.final_results.append(ModelResult(
            model_name=model_name, feature_name=feature_name,
            feat_time_train=feat_time_train, feat_time_test=feat_time_test,
            train_time=train_time, test_time=test_time,
            accuracy=accuracy, balanced_accuracy=balanced_accuracy,
            f1_macro=f1_macro, cv_score=cv_score,
            best_params=best_params, per_class_accuracy=per_class_accuracy
        ))
    
    def get_feature_times(self, feature_name: str) -> Tuple[float, float]:
        """Returns train and test time for a feature extraction."""
        if feature_name in self.feature_extraction:
            stats = self.feature_extraction[feature_name]
            return stats.train_time, stats.test_time
        return 0.0, 0.0
    
    def print_detailed_runtime(self):
        """Prints detailed runtime overview."""
        print(f"\n{'='*100}")
        print("DETAILED RUNTIME ANALYSIS")
        print(f"{'='*100}")
        
        # Feature Extraction
        print(f"\n{'─'*100}")
        print("FEATURE EXTRACTION (Train vs Test)")
        print(f"{'─'*100}")
        print(f"{'Feature':<35} {'Train(s)':>10} {'Test(s)':>10} {'Total(s)':>10} "
              f"{'Train/Sample(ms)':>16} {'Test/Sample(ms)':>16}")
        print("-" * 100)
        
        for name, stats in self.feature_extraction.items():
            print(f"{name:<35} {stats.train_time:>10.2f} {stats.test_time:>10.2f} "
                  f"{stats.total_time:>10.2f} {stats.train_per_sample*1000:>16.4f} "
                  f"{stats.test_per_sample*1000:>16.4f}")
        
        # Model Training
        print(f"\n{'─'*100}")
        print("MODEL TRAINING & PREDICTION (Mean CV Train Time vs Test Prediction)")
        print(f"{'─'*100}")
        print(f"{'Model':<45} {'MeanTrain(s)':>12} {'Test(s)':>10} "
              f"{'Train/Sample(ms)':>16} {'Test/Sample(ms)':>16}")
        print("-" * 100)
        
        for name, stats in self.model_training.items():
            print(f"{name:<45} {stats.train_time:>12.4f} {stats.test_time:>10.4f} "
                  f"{stats.train_per_sample*1000:>16.6f} "
                  f"{stats.test_per_sample*1000:>16.6f}")
    
    def print_final_summary(self):
        """Prints final summary table (best LR per feature, RF per feature)."""
        if not self.final_results:
            return
        
        # Group by feature and model type (LR vs RF)
        best_models = {}
        for r in self.final_results:
            is_lr = r.model_name.startswith('LR_')
            model_type = 'LR' if is_lr else 'RF'
            key = (model_type, r.feature_name)
            
            if key not in best_models or r.accuracy > best_models[key].accuracy:
                best_models[key] = r
        
        sorted_results = sorted(best_models.values(), 
                                key=lambda x: (x.feature_name, x.model_name))
            
        print(f"\n{'='*140}")
        print("FINAL PERFORMANCE SUMMARY (Best LR & RF per Feature)")
        print(f"{'='*140}")
        print(f"{'Model':<25} {'FeatTrain(s)':>12} {'FeatTest(s)':>12} {'ModelTrain(s)':>13} "
              f"{'ModelTest(s)':>12} {'Accuracy':>10} {'Bal.Acc':>10} {'CV Score':>10}")
        print("-" * 140)
        
        for r in sorted_results:
            model_feat = f"{r.model_name}+{r.feature_name[:8]}"
            print(f"{model_feat:<25} {r.feat_time_train:>12.4f} {r.feat_time_test:>12.4f} "
                  f"{r.train_time:>13.4f} {r.test_time:>12.4f} "
                  f"{r.accuracy:>10.4f} {r.balanced_accuracy:>10.4f} {r.cv_score:>10.4f}")
    
    def save_best_summary_csv(self, out_path: Path):
        import csv
        if not self.final_results:
            return
        
        # Group by feature and model type (LR vs RF)
        best_models = {}
        for r in self.final_results:
            # Determine if LR or RF
            is_lr = r.model_name.startswith('LR_')
            model_type = 'LR' if is_lr else 'RF'
            key = (model_type, r.feature_name)
            
            # Keep only the best model per (model type, feature) based on Accuracy
            if key not in best_models or r.accuracy > best_models[key].accuracy:
                best_models[key] = r
        
        # Sort by feature, then model type
        sorted_results = sorted(best_models.values(), 
                                key=lambda x: (x.feature_name, x.model_name))
        
        fieldnames = [
            'model', 'feature', 'feat_time_train_s', 'feat_time_test_s',
            'model_train_time_s', 'model_test_time_s',
            'accuracy', 'balanced_accuracy', 'f1_macro', 'cv_score', 'best_params'
        ]
        
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted_results:
                writer.writerow({
                    'model': r.model_name,
                    'feature': r.feature_name,
                    'feat_time_train_s': f"{r.feat_time_train:.4f}",
                    'feat_time_test_s': f"{r.feat_time_test:.4f}",
                    'model_train_time_s': f"{r.train_time:.4f}",
                    'model_test_time_s': f"{r.test_time:.4f}",
                    'accuracy': f"{r.accuracy:.4f}",
                    'balanced_accuracy': f"{r.balanced_accuracy:.4f}",
                    'f1_macro': f"{r.f1_macro:.4f}",
                    'cv_score': f"{r.cv_score:.4f}",
                    'best_params': str(r.best_params) if r.best_params else '-',
                })
        print(f"  Saved best summary ({len(sorted_results)} models): {out_path.name}")
    

    
    def to_dict(self) -> Dict:
        """Exports all data as a dictionary."""
        return {
            'feature_extraction': {
                name: {
                    'train_time_s': s.train_time,
                    'test_time_s': s.test_time,
                    'total_time_s': s.total_time,
                    'train_samples': s.train_samples,
                    'test_samples': s.test_samples,
                    'train_per_sample_ms': s.train_per_sample * 1000,
                    'test_per_sample_ms': s.test_per_sample * 1000,
                } for name, s in self.feature_extraction.items()
            },
            'model_training': {
                name: {
                    'mean_train_time_s': s.train_time,
                    'test_time_s': s.test_time,
                    'total_time_s': s.total_time,
                    'train_samples': s.train_samples,
                    'test_samples': s.test_samples,
                    'train_per_sample_ms': s.train_per_sample * 1000,
                    'test_per_sample_ms': s.test_per_sample * 1000,
                } for name, s in self.model_training.items()
            },
            'final_summary': [
                {
                    'model': r.model_name,
                    'feature': r.feature_name,
                    'feat_time_train_s': r.feat_time_train,
                    'feat_time_test_s': r.feat_time_test,
                    'model_train_time_s': r.train_time,
                    'model_test_time_s': r.test_time,
                    'accuracy': r.accuracy,
                    'balanced_accuracy': r.balanced_accuracy,
                    'f1_macro': r.f1_macro,
                    'cv_score': r.cv_score,
                    'best_params': r.best_params,
                } for r in self.final_results
            ],
            'timing_notes': {
                'feature_extraction': 'Train and test times measured directly (wall clock)',
                'model_train_time': 'Mean fit time from CV (averaged over folds)',
                'model_test_time': 'Single prediction on test set after CV (wall clock)',
            }
        }

# Global runtime tracker
runtime_tracker = RuntimeTracker()

# =============================================================================
# TIMER / STOPWATCH
# =============================================================================
class Timer:
    """Simple timer/stopwatch for tracking execution times."""
    def __init__(self):
        self.start_time = None
        self.lap_times: Dict[str, float] = {}
        self._lap_start: Dict[str, float] = {}

    def start(self):
        self.start_time = time.time()
        print(f"\n⏱️  Timer started at {datetime.now().strftime('%H:%M:%S')}")

    def lap_start(self, name: str):
        self._lap_start[name] = time.time()

    def lap_end(self, name: str, print_result: bool = True) -> float:
        if name not in self._lap_start:
            return 0.0
        elapsed = time.time() - self._lap_start[name]
        self.lap_times[name] = elapsed
        if print_result:
            print(f"⏱️  [{name}] completed in {self._format_time(elapsed)}")
        return elapsed

    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0

    def stop(self) -> float:
        total = self.elapsed()
        print(f"\n{'='*60}")
        print(f"⏱️  TOTAL EXECUTION TIME: {self._format_time(total)}")
        print(f"{'='*60}")
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_seconds': self.elapsed(),
            'total_formatted': self._format_time(self.elapsed()),
            'laps': {k: {'seconds': v, 'formatted': self._format_time(v)} 
                     for k, v in self.lap_times.items()}
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            mins, secs = divmod(seconds, 60)
            return f"{int(mins)}m {secs:.1f}s"
        else:
            hours, remainder = divmod(seconds, 3600)
            mins, secs = divmod(remainder, 60)
            return f"{int(hours)}h {int(mins)}m {secs:.0f}s"

timer = Timer()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def set_seed(seed: int = 42):
    """Sets seed for reproducibility (Numpy, Random, Torch)."""
    random.seed(seed)                          
    os.environ['PYTHONHASHSEED'] = str(seed)   
    np.random.seed(seed)                       
    
    # fixes the seed for the loaders
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def slugify(s: str, max_len: int = 140) -> str:
    import re
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-_\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]

# =============================================================================
# DATA LOADING
# =============================================================================
def loader_to_numpy(loader, max_items=None):
    Xs, ys = [], []
    n = 0
    for xb, yb in loader:
        Xs.append(xb.numpy())
        ys.append(yb.numpy())
        n += xb.shape[0]
        if max_items is not None and n >= max_items:
            break
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_items is not None:
        X, y = X[:max_items], y[:max_items]
    return X, y

def load_dataset(dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    print(f"\n{'='*60}")
    print(f"Loading {dataset.upper()} dataset...")
    print(f"{'='*60}")
    timer.lap_start(f"Data Loading ({dataset})")

    if dataset == "cifar10":
        from data.cifar_loader import get_cifar10_dataloaders
        train_loader, test_loader = get_cifar10_dataloaders(
            root=DATA_ROOT, batch_size=128, normalize=False, img_size=(64, 64)
        )
        class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
    elif dataset == "gtsrb":
        from data.gtsrb_loader import get_gtsrb_dataloaders
        train_loader, test_loader = get_gtsrb_dataloaders(
            root=DATA_ROOT, batch_size=128, normalize=False, img_size=(64, 64)
        )
        class_names = [str(i) for i in range(43)]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    Xtr, ytr = loader_to_numpy(train_loader, MAX_TRAIN_SAMPLES)
    Xte, yte = loader_to_numpy(test_loader, MAX_TEST_SAMPLES)
    timer.lap_end(f"Data Loading ({dataset})")
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")
    print(f"Classes: {len(class_names)}")
    return Xtr, ytr, Xte, yte, class_names

# =============================================================================
# FEATURE EXTRACTION - With separate train/test timing
# =============================================================================
def _extract_color_hist_single(X: np.ndarray, bins: int) -> np.ndarray:
    """Extracts Color Histogram for an array of images."""
    feats = []
    for img_chw in X:
        img = (img_chw * 255).astype(np.uint8)
        img_hwc = np.transpose(img, (1, 2, 0))
        img_bgr = img_hwc[:, :, ::-1].copy()
        hist_3d = cv2.calcHist([img_bgr], [0, 1, 2], None,
                               [bins, bins, bins],
                               [0, 256, 0, 256, 0, 256])
        f = hist_3d.flatten().astype(np.float32)
        f /= (f.sum() + 1e-8)
        feats.append(f)
    return np.vstack(feats)

def extract_color_histogram_3d(Xtr: np.ndarray, Xte: np.ndarray, 
                                bins: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 3D color histogram features with separate timing."""
    print(f"\nExtracting 3D Color Histogram (bins={bins})...")
    feature_name = f"ColorHist_bins{bins}"
    
    # Train extraction
    print(f"  Processing TRAIN set ({len(Xtr)} images)...")
    train_start = time.time()
    X_train_feat = _extract_color_hist_single(Xtr, bins)
    train_time = time.time() - train_start
    
    # Test extraction
    print(f"  Processing TEST set ({len(Xte)} images)...")
    test_start = time.time()
    X_test_feat = _extract_color_hist_single(Xte, bins)
    test_time = time.time() - test_start
    
    # Store times
    runtime_tracker.add_feature_time(
        feature_name, train_time, test_time, len(Xtr), len(Xte)
    )
    
    print(f"  ✓ Train: {train_time:.2f}s ({train_time/len(Xtr)*1000:.4f}ms/sample)")
    print(f"  ✓ Test:  {test_time:.2f}s ({test_time/len(Xte)*1000:.4f}ms/sample)")
    print(f"  Feature dim: {X_train_feat.shape[1]}")
    
    return X_train_feat, X_test_feat

def extract_sift_bovw(Xtr: np.ndarray, Xte: np.ndarray, 
                      n_clusters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Extract SIFT + Bag of Visual Words features with separate timing."""
    print(f"\nExtracting SIFT + BoVW (clusters={n_clusters})...")
    feature_name = "SIFT_BoVW"
    
    from models.sift_bovw import BOV
    bov = BOV(no_clusters=n_clusters)
    
    # Train: Learn vocabulary + extract features
    print(f"  Training vocabulary on {len(Xtr)} images...")
    train_start = time.time()
    X_train_feats = bov.compute_train_features(Xtr)
    train_time = time.time() - train_start
    
    # Test: Extract features only (vocabulary already learned)
    print(f"  Encoding TEST set ({len(Xte)} images)...")
    test_start = time.time()
    X_test_feats = bov.compute_test_features(Xte)
    test_time = time.time() - test_start
    
    # Store times
    runtime_tracker.add_feature_time(
        feature_name, train_time, test_time, len(Xtr), len(Xte)
    )
    
    print(f"  ✓ Train: {train_time:.2f}s ({train_time/len(Xtr)*1000:.4f}ms/sample)")
    print(f"  ✓ Test:  {test_time:.2f}s ({test_time/len(Xte)*1000:.4f}ms/sample)")
    print(f"  Feature dim: {X_train_feats.shape[1]}")
    
    return X_train_feats, X_test_feats

# =============================================================================
# EVALUATION & PLOTTING
# =============================================================================
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                          title: str, out_path: Path, dpi: int = 150):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    if len(class_names) <= 20:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot_per_class_recall(cm: np.ndarray, class_names: List[str],
                          title: str, out_path: Path, dpi: int = 150):
    row_sums = cm.sum(axis=1)
    recalls = np.where(row_sums > 0, np.diag(cm) / row_sums, 0.0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(len(recalls)), recalls, color='steelblue')
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("Class index")
    ax.set_ylabel("Recall")
    #ax.axhline(y=np.mean(recalls), color='red', linestyle='--', 
    #           label=f'Mean: {np.mean(recalls):.3f}')
    #ax.legend()
    if len(class_names) <= 20:
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot_runtime_comparison(runtime_data: Dict, out_path: Path):
    """Creates visualization of runtime analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Feature Extraction
    ax1 = axes[0]
    feat_names = list(runtime_data['feature_extraction'].keys())
    if feat_names:
        train_times = [runtime_data['feature_extraction'][n]['train_time_s'] for n in feat_names]
        test_times = [runtime_data['feature_extraction'][n]['test_time_s'] for n in feat_names]
        
        x = np.arange(len(feat_names))
        width = 0.35
        ax1.bar(x - width/2, train_times, width, label='Train', color='steelblue')
        ax1.bar(x + width/2, test_times, width, label='Test', color='coral')
        ax1.set_xlabel('Feature Type')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Feature Extraction Runtime')
        ax1.set_xticks(x)
        ax1.set_xticklabels(feat_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # Model Training
    ax2 = axes[1]
    model_names = list(runtime_data['model_training'].keys())
    if model_names:
        train_times = [runtime_data['model_training'][n]['mean_train_time_s'] for n in model_names]
        test_times = [runtime_data['model_training'][n]['test_time_s'] for n in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        ax2.bar(x - width/2, train_times, width, label='Train', color='steelblue')
        ax2.bar(x + width/2, test_times, width, label='Test', color='coral')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Model Training & Test Runtime')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=7)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved runtime plot: {out_path.name}")

def plot_runtime_per_sample(runtime_data: Dict, out_path: Path):
    """Creates visualization of runtime per sample."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Feature Extraction per sample
    ax1 = axes[0]
    feat_names = list(runtime_data['feature_extraction'].keys())
    if feat_names:
        train_ps = [runtime_data['feature_extraction'][n]['train_per_sample_ms'] for n in feat_names]
        test_ps = [runtime_data['feature_extraction'][n]['test_per_sample_ms'] for n in feat_names]
        
        x = np.arange(len(feat_names))
        width = 0.35
        ax1.bar(x - width/2, train_ps, width, label='Train', color='steelblue')
        ax1.bar(x + width/2, test_ps, width, label='Test', color='coral')
        ax1.set_xlabel('Feature Type')
        ax1.set_ylabel('Time per Sample (ms)')
        ax1.set_title('Feature Extraction - Time per Sample')
        ax1.set_xticks(x)
        ax1.set_xticklabels(feat_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # Model per sample
    ax2 = axes[1]
    model_names = list(runtime_data['model_training'].keys())
    if model_names:
        train_ps = [runtime_data['model_training'][n]['train_per_sample_ms'] for n in model_names]
        test_ps = [runtime_data['model_training'][n]['test_per_sample_ms'] for n in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        ax2.bar(x - width/2, train_ps, width, label='Train', color='steelblue')
        ax2.bar(x + width/2, test_ps, width, label='Test', color='coral')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Time per Sample (ms)')
        ax2.set_title('Model - Time per Sample')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=7)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved per-sample runtime plot: {out_path.name}")

def save_all_models_csv(all_models_results: List[Dict], out_path: Path):
    import csv
    if not all_models_results:
        return
    fieldnames = ['rank', 'mean_test_score', 'std_test_score', 'mean_train_score', 
                  'std_train_score', 'mean_fit_time', 'std_fit_time']
    for key in all_models_results[0].keys():
        if key.startswith('split') and key not in fieldnames:
            fieldnames.append(key)
    param_keys = list(all_models_results[0]['params'].keys())
    fieldnames.extend(param_keys)

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_models_results:
            row = {k: result.get(k, '') for k in fieldnames if k not in param_keys}
            for pk in param_keys:
                row[pk] = result['params'].get(pk, '')
            writer.writerow(row)
    print(f"  Saved all {len(all_models_results)} model results to: {out_path.name}")

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   class_names: List[str]) -> Tuple[Dict, float, Dict[str, float], np.ndarray]:
    """Evaluate model and return metrics + test time + per-class accuracy + confusion matrix."""
    start = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    overall_acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Compute per-class accuracy (recall per class)
    row_sums = cm.sum(axis=1)
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        if row_sums[i] > 0:
            per_class_acc[class_name] = float(cm[i, i] / row_sums[i])
        else:
            per_class_acc[class_name] = 0.0

    metrics = {
        'overall_accuracy': float(overall_acc),
        'balanced_accuracy': float(bal_acc),
        'f1_macro': float(f1),
    }
    
    return metrics, test_time, per_class_acc, cm


def save_best_model_artifacts(out_dir: Path, title: str, class_names: List[str],
                              metrics: Dict, cm: np.ndarray, per_class_acc: Dict,
                              train_time: float, test_time: float, cv_score: float):
    """Save artifacts for the best model (plots, per_class_accuracy, metrics)."""
    
    metrics_full = {
        **metrics,
        'train_time_seconds': float(train_time),
        'test_time_seconds': float(test_time),
        'cv_score': float(cv_score),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_full, indent=2))
    
    # Save per-class accuracy
    per_class_data = {
        'overall_accuracy': metrics['overall_accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'per_class_accuracy': per_class_acc
    }
    (out_dir / "per_class_accuracy.json").write_text(json.dumps(per_class_data, indent=2))
    
    # Generate classification report
    # Reconstruct y_test and y_pred from confusion matrix for report
    # Actually we need to pass these - let's skip the report.txt for now or compute from cm
    
    # Plots
    plot_confusion_matrix(cm, class_names, f"Confusion Matrix - {title}", 
                          out_dir / "confusion_matrix.png")
    plot_per_class_recall(cm, class_names, f"Per-class Recall - {title}",
                          out_dir / "per_class_recall.png")

# =============================================================================
# MODEL TRAINING - With detailed timing
# =============================================================================
def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              class_names: List[str], feature_name: str,
                              dataset: str, base_dir: Path):
    print(f"\n{'='*60}")
    print("Training Logistic Regression models...")
    print(f"{'='*60}")

    configs = [
        {'name': 'no_penalty', 'penalty': None, 'solver': 'saga'},
        {'name': 'l1_lasso', 'penalty': 'l1', 'solver': 'saga', 'C_values': LR_C_VALUES},
        {'name': 'l2_ridge', 'penalty': 'l2', 'solver': 'saga', 'C_values': LR_C_VALUES},
        {'name': 'elasticnet', 'penalty': 'elasticnet', 'solver': 'saga', 
         'C_values': LR_C_VALUES, 'l1_ratio': [0.5]}
    ]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    all_results = {}
    
    # Get feature extraction times for this feature
    feat_time_train, feat_time_test = runtime_tracker.get_feature_times(feature_name)
    
    # Create output directory for all LR models with this feature
    run_dir = base_dir / f"LogisticRegression_{feature_name}_{dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Store results for all penalties
    all_penalty_results = []
    all_confusion_matrices = {}
    all_cv_results = {}

    for cfg in configs:
        print(f"\n--- {cfg['name'].upper()} ---")
        model_key = f"LR_{cfg['name']}_{feature_name}"

        if cfg['penalty'] is None:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(penalty=None, solver='saga',
                                           max_iter=1000, random_state=42, n_jobs=-1))
            ])
            param_grid = {}
        else:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(penalty=cfg['penalty'], solver=cfg['solver'],
                                           max_iter=1000, random_state=42, n_jobs=-1))
            ])
            param_grid = {'clf__C': cfg['C_values']}
            if cfg['penalty'] == 'elasticnet':
                param_grid['clf__l1_ratio'] = cfg.get('l1_ratio', [0.5])

        # Training
        if param_grid:
            grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=CV_SCORING,
                                return_train_score=True, n_jobs=-1, verbose=1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            
            best_idx = grid.best_index_
            mean_train_time = float(grid.cv_results_['mean_fit_time'][best_idx])
            best_params = grid.best_params_
            cv_score = float(grid.best_score_)
            
            all_models_results = []
            for i in range(len(grid.cv_results_['params'])):
                model_result = {
                    'rank': int(grid.cv_results_['rank_test_score'][i]),
                    'params': grid.cv_results_['params'][i],
                    'mean_test_score': float(grid.cv_results_['mean_test_score'][i]),
                    'std_test_score': float(grid.cv_results_['std_test_score'][i]),
                    'mean_train_score': float(grid.cv_results_['mean_train_score'][i]),
                    'std_train_score': float(grid.cv_results_['std_train_score'][i]),
                    'mean_fit_time': float(grid.cv_results_['mean_fit_time'][i]),
                    'std_fit_time': float(grid.cv_results_['std_fit_time'][i]),
                }
                for fold in range(CV_FOLDS):
                    model_result[f'split{fold}_test_score'] = float(
                        grid.cv_results_[f'split{fold}_test_score'][i])
                all_models_results.append(model_result)
            all_models_results.sort(key=lambda x: x['rank'])
            cv_results_data = {
                'best_score': grid.best_score_,
                'best_params': grid.best_params_,
                'all_models': all_models_results,
            }
            print(f"  Best params: {grid.best_params_}")
            print(f"  Best CV score: {grid.best_score_:.4f}")
        else:
            print(f"  Running {CV_FOLDS}-fold CV...")
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, 
                                        scoring=CV_SCORING, n_jobs=-1)
            cv_score = float(np.mean(cv_scores))
            print(f"  CV score: {cv_score:.4f} (+/- {np.std(cv_scores):.4f})")
            
            train_start = time.time()
            pipe.fit(X_train, y_train)
            mean_train_time = time.time() - train_start
            best_model = pipe
            cv_results_data = {
                'best_score': cv_score,
                'cv_scores': cv_scores.tolist(),
                'cv_std': float(np.std(cv_scores)),
            }
            all_models_results = None
            best_params = {}

        # Evaluate model
        metrics, test_time, per_class_acc, cm = evaluate_model(
            best_model, X_test, y_test, class_names
        )
        
        # Store confusion matrix
        all_confusion_matrices[cfg['name']] = {
            'confusion_matrix': cm.tolist(),
            'params': best_params,
            'accuracy': metrics['overall_accuracy'],
            'cv_score': cv_score,
        }
        
        # Store CV results
        all_cv_results[cfg['name']] = cv_results_data
        
        # Store for comparison
        penalty_result = {
            'name': cfg['name'],
            'model': best_model,
            'metrics': metrics,
            'test_time': test_time,
            'train_time': mean_train_time,
            'per_class_acc': per_class_acc,
            'cm': cm,
            'cv_score': cv_score,
            'best_params': best_params,
            'all_models_results': all_models_results,
        }
        all_penalty_results.append(penalty_result)
        
        # Store in Runtime Tracker
        runtime_tracker.add_model_time(
            model_key, mean_train_time, test_time, len(X_train), len(X_test)
        )
        
        # Add to final results (for all models)
        runtime_tracker.add_final_result(
            model_name=f"LR_{cfg['name']}",
            feature_name=feature_name,
            feat_time_train=feat_time_train,
            feat_time_test=feat_time_test,
            train_time=mean_train_time,
            test_time=test_time,
            accuracy=metrics['overall_accuracy'],
            balanced_accuracy=metrics['balanced_accuracy'],
            f1_macro=metrics['f1_macro'],
            cv_score=cv_score,
            best_params=best_params,
            per_class_accuracy=per_class_acc
        )

        all_results[cfg['name']] = metrics
        all_results[cfg['name']]['cv_mean_test_score'] = cv_score
        all_results[cfg['name']]['train_time_seconds'] = mean_train_time
        all_results[cfg['name']]['test_time_seconds'] = test_time
        if cv_results_data and 'all_models' in cv_results_data and cv_results_data['all_models']:
            all_results[cfg['name']]['cv_std_test_score'] = cv_results_data['all_models'][0]['std_test_score']
        elif cv_results_data and 'cv_std' in cv_results_data:
            all_results[cfg['name']]['cv_std_test_score'] = cv_results_data['cv_std']
        else:
            all_results[cfg['name']]['cv_std_test_score'] = 0
        all_results[cfg['name']]['rank'] = 1 if param_grid else '-'

        print(f"  ✓ Mean Train Time (CV): {mean_train_time:.4f}s, Test Time: {test_time:.4f}s")
        print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")

    # Find best model by accuracy
    best_penalty_result = max(all_penalty_results, key=lambda x: x['metrics']['overall_accuracy'])
    print(f"\n  >>> Best LR model: {best_penalty_result['name']} with accuracy {best_penalty_result['metrics']['overall_accuracy']:.4f}")
    
    # Save config for all penalties
    config = {
        'model': 'LogisticRegression',
        'feature': feature_name,
        'dataset': dataset,
        'cv_folds': CV_FOLDS,
        'scoring': CV_SCORING,
        'best_penalty': best_penalty_result['name'],
        'best_params': best_penalty_result['best_params'],
        'penalties_tested': [cfg['name'] for cfg in configs],
        'created_at': datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    
    # Save CV results for all penalties
    #(run_dir / "cv_results.json").write_text(json.dumps(all_cv_results, indent=2, default=str))
    
    # Save all confusion matrices
    (run_dir / "all_confusion_matrices.json").write_text(json.dumps(all_confusion_matrices, indent=2))
    
    # Save all models results CSV (combine all penalties with consistent fields)
    all_models_combined = []
    for penalty_result in all_penalty_results:
        if penalty_result['all_models_results']:
            # GridSearchCV results - multiple hyperparameter combinations
            for model_res in penalty_result['all_models_results']:
                combined_entry = {
                    'penalty': penalty_result['name'],
                    'rank': model_res.get('rank', '-'),
                    'params': str(model_res.get('params', {})),
                    'cv_mean_test_score': model_res.get('mean_test_score', 0),
                    'cv_std_test_score': model_res.get('std_test_score', 0),
                    'cv_mean_train_score': model_res.get('mean_train_score', 0),
                    'cv_std_train_score': model_res.get('std_train_score', 0),
                    'mean_fit_time': model_res.get('mean_fit_time', 0),
                    'std_fit_time': model_res.get('std_fit_time', 0),
                }
                # Add fold scores
                for fold in range(CV_FOLDS):
                    fold_key = f'split{fold}_test_score'
                    combined_entry[fold_key] = model_res.get(fold_key, 0)
                all_models_combined.append(combined_entry)
        else:
            # No GridSearchCV (e.g., no_penalty) - single entry
            combined_entry = {
                'penalty': penalty_result['name'],
                'rank': 1,
                'params': str(penalty_result['best_params']),
                'cv_mean_test_score': penalty_result['cv_score'],
                'cv_std_test_score': all_cv_results.get(penalty_result['name'], {}).get('cv_std', 0),
                'cv_mean_train_score': 0,  # Not available for cross_val_score
                'cv_std_train_score': 0,
                'mean_fit_time': penalty_result['train_time'],
                'std_fit_time': 0,
            }
            # Add fold scores if available
            cv_scores = all_cv_results.get(penalty_result['name'], {}).get('cv_scores', [])
            for fold in range(CV_FOLDS):
                fold_key = f'split{fold}_test_score'
                combined_entry[fold_key] = cv_scores[fold] if fold < len(cv_scores) else 0
            all_models_combined.append(combined_entry)
    
    if all_models_combined:
        import csv
        fieldnames = ['penalty', 'rank', 'params', 'cv_mean_test_score', 'cv_std_test_score',
                      'cv_mean_train_score', 'cv_std_train_score', 'mean_fit_time', 'std_fit_time']
        fieldnames += [f'split{fold}_test_score' for fold in range(CV_FOLDS)]
        
        with open(run_dir / "results.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_models_combined:
                writer.writerow(row)
    
    # Save artifacts only for the best model (plots, per_class_accuracy)
    title = f"Logistic Regression - {feature_name} - {dataset}"
    save_best_model_artifacts(
        run_dir, title, class_names,
        best_penalty_result['metrics'],
        best_penalty_result['cm'],
        best_penalty_result['per_class_acc'],
        best_penalty_result['train_time'],
        best_penalty_result['test_time'],
        best_penalty_result['cv_score']
    )
    
    print(f"  Saved to: {run_dir}")

    return all_results

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        class_names: List[str], feature_name: str,
                        dataset: str, base_dir: Path):
    print(f"\n{'='*60}")
    print("Training Random Forest...")
    print(f"{'='*60}")
    print(f"Parameter grid: {RF_PARAM_GRID}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Training with GridSearchCV
    grid = GridSearchCV(base_model, RF_PARAM_GRID, cv=cv, scoring=CV_SCORING,
                        return_train_score=True, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    
    # Get mean_fit_time for best model
    best_idx = grid.best_index_
    mean_train_time = float(grid.cv_results_['mean_fit_time'][best_idx])
    best_params = grid.best_params_
    cv_score = float(grid.best_score_)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")

    all_models_results = []
    all_confusion_matrices = {}
    
    for i in range(len(grid.cv_results_['params'])):
        model_result = {
            'rank': int(grid.cv_results_['rank_test_score'][i]),
            'params': grid.cv_results_['params'][i],
            'mean_test_score': float(grid.cv_results_['mean_test_score'][i]),
            'std_test_score': float(grid.cv_results_['std_test_score'][i]),
            'mean_train_score': float(grid.cv_results_['mean_train_score'][i]),
            'std_train_score': float(grid.cv_results_['std_train_score'][i]),
            'mean_fit_time': float(grid.cv_results_['mean_fit_time'][i]),
            'std_fit_time': float(grid.cv_results_['std_fit_time'][i]),
        }
        for fold in range(CV_FOLDS):
            model_result[f'split{fold}_test_score'] = float(
                grid.cv_results_[f'split{fold}_test_score'][i])
        all_models_results.append(model_result)
    all_models_results.sort(key=lambda x: x['rank'])

    cv_results = {
        'best_score': grid.best_score_,
        'best_params': grid.best_params_,
        'all_models': all_models_results,
    }

    run_dir = base_dir / f"RandomForest_{feature_name}_{dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'model': 'RandomForest',
        'feature': feature_name,
        'dataset': dataset,
        'cv_folds': CV_FOLDS,
        'scoring': CV_SCORING,
        'param_grid': {k: [str(v) for v in vals] for k, vals in RF_PARAM_GRID.items()},
        'best_params': {k: str(v) for k, v in grid.best_params_.items()},
        'created_at': datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    # (run_dir / "cv_results.json").write_text(json.dumps(cv_results, indent=2, default=str))
    save_all_models_csv(all_models_results, run_dir / "results.csv")

    # Evaluate best model
    best_metrics, test_time, per_class_acc, cm = evaluate_model(
        grid.best_estimator_, X_test, y_test, class_names
    )
    
    # Save confusion matrix for best model
    all_confusion_matrices['best'] = {
        'confusion_matrix': cm.tolist(),
        'params': best_params,
        'accuracy': best_metrics['overall_accuracy'],
        'cv_score': cv_score,
    }
    (run_dir / "all_confusion_matrices.json").write_text(json.dumps(all_confusion_matrices, indent=2))

    # Save artifacts for best model
    title = f"Random Forest - {feature_name} - {dataset}"
    save_best_model_artifacts(
        run_dir, title, class_names,
        best_metrics, cm, per_class_acc,
        mean_train_time, test_time, cv_score
    )

    # Store in Runtime Tracker (mean train time from CV, test time directly measured)
    model_key = f"RF_best_{feature_name}"
    runtime_tracker.add_model_time(
        model_key, mean_train_time, test_time, len(X_train), len(X_test)
    )
    
    # Get feature extraction times for this feature
    feat_time_train, feat_time_test = runtime_tracker.get_feature_times(feature_name)
    
    # Add to final results
    runtime_tracker.add_final_result(
        model_name="RF",
        feature_name=feature_name,
        feat_time_train=feat_time_train,
        feat_time_test=feat_time_test,
        train_time=mean_train_time,
        test_time=test_time,
        accuracy=best_metrics['overall_accuracy'],
        balanced_accuracy=best_metrics['balanced_accuracy'],
        f1_macro=best_metrics['f1_macro'],
        cv_score=cv_score,
        best_params=best_params,
        per_class_accuracy=per_class_acc
    )

    print(f"  ✓ Mean Train Time (CV): {mean_train_time:.4f}s, Test Time: {test_time:.4f}s")
    print(f"Best Model Accuracy: {best_metrics['overall_accuracy']:.4f}")
    print(f"Best Model Balanced Acc: {best_metrics['balanced_accuracy']:.4f}")
    print(f"Saved to: {run_dir}")

    all_results = {}
    for model_result in all_models_results:
        params = model_result['params']
        model_name = f"n{params.get('n_estimators', '')}_d{params.get('max_depth', 'None')}"
        all_results[model_name] = {
            'overall_accuracy': model_result['mean_test_score'],
            'balanced_accuracy': model_result['mean_test_score'],
            'f1_macro': model_result['mean_test_score'],
            'train_time_seconds': model_result['mean_fit_time'],
            'cv_mean_test_score': model_result['mean_test_score'],
            'cv_std_test_score': model_result['std_test_score'],
            'rank': model_result['rank'],
            'params': params,
        }
    return all_results

# =============================================================================
# MAIN
# =============================================================================
def print_summary(all_results: Dict, dataset: str):
    print(f"\n{'='*100}")
    print(f"MODEL PERFORMANCE SUMMARY - {dataset.upper()}")
    print(f"{'='*100}")
    print(f"{'Model':<55} {'CV_Score':>10} {'Std':>10} {'Rank':>6} {'Train(s)':>10}")
    print("-" * 100)

    lr_results = {k: v for k, v in all_results.items() if k.startswith('LR_')}
    rf_results = {k: v for k, v in all_results.items() if k.startswith('RF_')}

    if lr_results:
        print("\n--- LOGISTIC REGRESSION ---")
        for name, metrics in sorted(lr_results.items()):
            cv_score = metrics.get('cv_mean_test_score', metrics.get('balanced_accuracy', 0))
            std = metrics.get('cv_std_test_score', 0)
            rank = metrics.get('rank', '-')
            print(f"{name:<55} {cv_score:>10.4f} {std:>10.4f} {str(rank):>6} "
                  f"{metrics['train_time_seconds']:>10.1f}")

    if rf_results:
        print("\n--- RANDOM FOREST ---")
        rf_sorted = sorted(rf_results.items(), key=lambda x: x[1].get('rank', 999))
        for name, metrics in rf_sorted:
            cv_score = metrics.get('cv_mean_test_score', metrics.get('balanced_accuracy', 0))
            std = metrics.get('cv_std_test_score', 0)
            rank = metrics.get('rank', '-')
            print(f"{name:<55} {cv_score:>10.4f} {std:>10.4f} {str(rank):>6} "
                  f"{metrics['train_time_seconds']:>10.1f}")

def main():
    set_seed(42)
    timer.start()

    print("\n" + "="*60)
    print("SHALLOW ML BASELINES FOR IMAGE CLASSIFICATION")
    print("Task 3.2.1 - Feature Extraction & Shallow Learning")
    print("="*60)

    # Dataset selection
    print("\nSelect dataset:")
    print("  1. CIFAR-10")
    print("  2. GTSRB")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == '1':
            dataset = 'cifar10'
            break
        elif choice == '2':
            dataset = 'gtsrb'
            break
        print("Invalid choice. Please enter 1 or 2.")

    # Model selection
    print("\nSelect model:")
    print("  1. Logistic Regression only")
    print("  2. Random Forest only")
    print("  3. Both")
    while True:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            model_choice = int(choice)
            break
        print("Invalid choice.")

    # Feature selection
    print("\nSelect features:")
    print("  1. Color Histogram only")
    print("  2. SIFT + BoVW only")
    print("  3. Both")
    while True:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            feature_choice = int(choice)
            break
        print("Invalid choice.")

    # Confirmation
    bins_info = COLOR_HIST_BINS if isinstance(COLOR_HIST_BINS, list) else [COLOR_HIST_BINS]
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset}")
    print(f"  Models: {'LR' if model_choice==1 else 'RF' if model_choice==2 else 'LR + RF'}")
    print(f"  Features: {'ColorHist' if feature_choice==1 else 'SIFT+BoVW' if feature_choice==2 else 'Both'}")
    if feature_choice in [1, 3]:
        print(f"  ColorHist bins: {bins_info}")
    print(f"  CV Folds: {CV_FOLDS}")

    while True:
        confirm = input("\nStart training? (y/n): ").strip().lower()
        if confirm == 'y':
            break
        elif confirm == 'n':
            print("Aborted.")
            return
        print("Please enter 'y' or 'n'.")

    # Load data
    Xtr, ytr, Xte, yte, class_names = load_dataset(dataset)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = Path("runs_ml") / dataset.upper() / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Extract features and train models
    if feature_choice in [1, 3]:  # Color Histogram
        print("\n" + "="*60)
        print("FEATURE: COLOR HISTOGRAM")
        print("="*60)
        bins_list = COLOR_HIST_BINS if isinstance(COLOR_HIST_BINS, list) else [COLOR_HIST_BINS]
        
        for bins in bins_list:
            print(f"\n>>> Processing Color Histogram bins={bins}")
            X_train_hist, X_test_hist = extract_color_histogram_3d(Xtr, Xte, bins=bins)
            feature_name = f"ColorHist_bins{bins}"

            if model_choice in [1, 3]:
                lr_results = train_logistic_regression(
                    X_train_hist, ytr, X_test_hist, yte, class_names,
                    feature_name, dataset, base_dir
                )
                for name, metrics in lr_results.items():
                    all_results[f"LR_{name}_{feature_name}"] = metrics

            if model_choice in [2, 3]:
                rf_results = train_random_forest(
                    X_train_hist, ytr, X_test_hist, yte, class_names,
                    feature_name, dataset, base_dir
                )
                for name, metrics in rf_results.items():
                    all_results[f"RF_{name}_{feature_name}"] = metrics

    if feature_choice in [2, 3]:  # SIFT + BoVW
        print("\n" + "="*60)
        print("FEATURE: SIFT + BAG OF VISUAL WORDS")
        print("="*60)
        X_train_bovw, X_test_bovw = extract_sift_bovw(Xtr, Xte, n_clusters=BOVW_N_CLUSTERS)

        if model_choice in [1, 3]:
            lr_results = train_logistic_regression(
                X_train_bovw, ytr, X_test_bovw, yte, class_names,
                "SIFT_BoVW", dataset, base_dir
            )
            for name, metrics in lr_results.items():
                all_results[f"LR_{name}_SIFT_BoVW"] = metrics

        if model_choice in [2, 3]:
            rf_results = train_random_forest(
                X_train_bovw, ytr, X_test_bovw, yte, class_names,
                "SIFT_BoVW", dataset, base_dir
            )
            for name, metrics in rf_results.items():
                all_results[f"RF_{name}_SIFT_BoVW"] = metrics

    # Print summaries
    print_summary(all_results, dataset)
    runtime_tracker.print_detailed_runtime()
    runtime_tracker.print_final_summary()

    # Save everything
    runtime_data = runtime_tracker.to_dict()
    
    # Generate runtime plots
    plot_runtime_comparison(runtime_data, base_dir / "runtime_comparison.png")
    plot_runtime_per_sample(runtime_data, base_dir / "runtime_per_sample.png")
    
    # Save best summary CSV
    runtime_tracker.save_best_summary_csv(base_dir / "best_summary.csv")

    summary_data = {
        'results': all_results,
        'runtime_analysis': runtime_data,
        'timing': timer.to_dict()
    }
    summary_path = base_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2))

    timer.stop()
    print(f"\nAll results saved to: {base_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"Best models summary: best_summary.csv")
    print(f"Per-class accuracy: <model_folder>/per_class_accuracy.json")
    print(f"Runtime plots: runtime_comparison.png, runtime_per_sample.png")

if __name__ == "__main__":
    main()
