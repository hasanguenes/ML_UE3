"""
Features:
- Color Histogram (3D)
- SIFT + Bag of Visual Words (BoVW)

Classifiers:
- Logistic Regression (no penalty, L1, L2, ElasticNet)
- Random Forest

Usage: python run_ml_models_with_gridsearch.py

Interactive prompts will guide you through dataset and model selection.
"""

import numpy as np
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 15, 25],  # or None
    'min_samples_split': [2, 5, 10],
    # Uncomment to add more parameters:
    # 'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['sqrt', 'log2', None],
    # 'bootstrap': [True, False],
}

# Logistic Regression C values (only used when penalty is not None)
LR_C_VALUES = [0.1, 1, 10]

# Cross-validation settings
CV_FOLDS = 10
CV_SCORING = 'accuracy'  # or 'balanced_accuracy', 'f1_macro'

# Feature extraction settings
COLOR_HIST_BINS = [8]  # [6, 8, 10]
BOVW_N_CLUSTERS = 100  # Number of visual words

# Data settings
DATA_ROOT = "data"
MAX_TRAIN_SAMPLES = None  # Set to int for debugging, None for full dataset
MAX_TEST_SAMPLES = None

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
        """Start the main timer."""
        self.start_time = time.time()
        print(f"\n⏱️  Timer started at {datetime.now().strftime('%H:%M:%S')}")
    
    def lap_start(self, name: str):
        """Start a named lap/section."""
        self._lap_start[name] = time.time()
    
    def lap_end(self, name: str, print_result: bool = True) -> float:
        """End a named lap/section and return elapsed time."""
        if name not in self._lap_start:
            return 0.0
        elapsed = time.time() - self._lap_start[name]
        self.lap_times[name] = elapsed
        if print_result:
            print(f"⏱️  [{name}] completed in {self._format_time(elapsed)}")
        return elapsed
    
    def elapsed(self) -> float:
        """Get total elapsed time since start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def stop(self) -> float:
        """Stop timer and print total time."""
        total = self.elapsed()
        print(f"\n{'='*60}")
        print(f"⏱️  TOTAL EXECUTION TIME: {self._format_time(total)}")
        print(f"{'='*60}")
        return total
    
    def summary(self):
        """Print summary of all recorded lap times."""
        if not self.lap_times:
            return
        print(f"\n{'='*60}")
        print("⏱️  TIME BREAKDOWN")
        print(f"{'='*60}")
        for name, elapsed in self.lap_times.items():
            pct = (elapsed / self.elapsed() * 100) if self.elapsed() > 0 else 0
            print(f"  {name:<40} {self._format_time(elapsed):>12} ({pct:>5.1f}%)")
        print(f"  {'─'*58}")
        print(f"  {'TOTAL':<40} {self._format_time(self.elapsed()):>12}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export timer data as dictionary."""
        return {
            'total_seconds': self.elapsed(),
            'total_formatted': self._format_time(self.elapsed()),
            'laps': {k: {'seconds': v, 'formatted': self._format_time(v)} 
                     for k, v in self.lap_times.items()}
        }
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            mins, secs = divmod(seconds, 60)
            return f"{int(mins)}m {secs:.1f}s"
        else:
            hours, remainder = divmod(seconds, 3600)
            mins, secs = divmod(remainder, 60)
            return f"{int(hours)}h {int(mins)}m {secs:.0f}s"

# Global timer instance
timer = Timer()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def set_seed(seed: int = 42):
    np.random.seed(seed)

def slugify(s: str, max_len: int = 140) -> str:
    import re
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-_\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]

def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    row_sums = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.where(row_sums > 0, np.diag(cm) / row_sums, 0.0)
    return float(np.mean(recalls))

# =============================================================================
# DATA LOADING
# =============================================================================

def loader_to_numpy(loader, max_items=None):
    """Convert PyTorch DataLoader to NumPy arrays."""
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
    """Load dataset and return train/test splits with class names."""
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
# FEATURE EXTRACTION
# =============================================================================

def extract_color_histogram_3d(X: np.ndarray, bins: int = 8) -> np.ndarray:
    """Extract 3D color histogram features."""
    print(f"\nExtracting 3D Color Histogram (bins={bins})...")
    
    timer.lap_start(f"ColorHist (bins={bins})")
    
    feats = []
    for i, img_chw in enumerate(X):
        img = (img_chw * 255).astype(np.uint8)
        img_hwc = np.transpose(img, (1, 2, 0))
        img_bgr = img_hwc[:, :, ::-1].copy()
        
        hist_3d = cv2.calcHist([img_bgr], [0, 1, 2], None,
                               [bins, bins, bins],
                               [0, 256, 0, 256, 0, 256])
        f = hist_3d.flatten().astype(np.float32)
        f /= (f.sum() + 1e-8)
        feats.append(f)
        
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(X)} images")
    
    timer.lap_end(f"ColorHist (bins={bins})")
    print(f"  Done.")
    return np.vstack(feats)

def extract_sift_bovw(Xtr: np.ndarray, Xte: np.ndarray, n_clusters: int = 100):
    """Extract SIFT + Bag of Visual Words features."""
    print(f"\nExtracting SIFT + BoVW (clusters={n_clusters})...")
    
    timer.lap_start("SIFT + BoVW")
    
    from models.sift_bovw import BOV
    bov = BOV(no_clusters=n_clusters)
    
    print("  Training vocabulary...")
    X_train_feats = bov.compute_train_features(Xtr)
    
    print("  Encoding test set...")
    X_test_feats = bov.compute_test_features(Xte)
    
    timer.lap_end("SIFT + BoVW")
    
    print(f"  Done.")
    print(f"  Train features: {X_train_feats.shape}, Test features: {X_test_feats.shape}")
    
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
    ax.axhline(y=np.mean(recalls), color='red', linestyle='--', label=f'Mean: {np.mean(recalls):.3f}')
    ax.legend()
    
    if len(class_names) <= 20:
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot_cv_results(cv_results: Dict, param_name: str, title: str, out_path: Path):
    """Plot cross-validation results for a single parameter."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    means = cv_results['mean_test_score']
    stds = cv_results['std_test_score']
    params = cv_results['params']
    
    x = range(len(means))
    ax.errorbar(x, means, yerr=stds, fmt='o-', capsize=3)
    ax.set_xlabel('Parameter combination')
    ax.set_ylabel(f'CV Score ({CV_SCORING})')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in params], rotation=45, ha='right', fontsize=6)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_all_models_csv(all_models_results: List[Dict], out_path: Path):
    """Save all model results to a readable CSV file."""
    import csv
    
    if not all_models_results:
        return
    
    # Get all keys from first result for header
    fieldnames = ['rank', 'mean_test_score', 'std_test_score', 'mean_train_score', 
                  'std_train_score', 'mean_fit_time', 'std_fit_time']
    
    # Add fold scores
    for key in all_models_results[0].keys():
        if key.startswith('split') and key not in fieldnames:
            fieldnames.append(key)
    
    # Add params at the end
    param_keys = list(all_models_results[0]['params'].keys())
    fieldnames.extend(param_keys)
    
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_models_results:
            row = {k: result.get(k, '') for k in fieldnames if k not in param_keys}
            # Flatten params
            for pk in param_keys:
                row[pk] = result['params'].get(pk, '')
            writer.writerow(row)
    
    print(f"  Saved all {len(all_models_results)} model results to: {out_path.name}")

def evaluate_and_save(model, X_test: np.ndarray, y_test: np.ndarray,
                      class_names: List[str], out_dir: Path, title: str,
                      cv_results: Optional[Dict] = None,
                      train_time: float = 0) -> Dict:
    """Evaluate model and save all artifacts."""
    
    # Predictions
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    overall_acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    report_txt = classification_report(
        y_test, y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4, zero_division=0
    )
    
    # Save metrics
    metrics = {
        'overall_accuracy': float(overall_acc),
        'balanced_accuracy': float(bal_acc),
        'f1_macro': float(f1),
        'train_time_seconds': float(train_time),
        'predict_time_seconds': float(predict_time),
    }
    
    if cv_results is not None:
        metrics['best_cv_score'] = float(cv_results.get('best_score', 0))
        metrics['cv_mean_fit_time'] = float(np.mean(cv_results.get('mean_fit_time', [0])))
    
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "report.txt").write_text(report_txt)
    
    # Save confusion matrix as numpy
    np.save(out_dir / "confusion_matrix.npy", cm)
    
    # Plots
    plot_confusion_matrix(cm, class_names, f"Confusion Matrix - {title}", 
                          out_dir / "confusion_matrix.png")
    plot_per_class_recall(cm, class_names, f"Per-class Recall - {title}",
                          out_dir / "per_class_recall.png")
    
    if cv_results is not None and 'params' in cv_results:
        plot_cv_results(cv_results, 'params', f"CV Results - {title}",
                        out_dir / "cv_results.png")
    
    return metrics

# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               class_names: List[str], feature_name: str,
                               dataset: str, base_dir: Path):
    """Train Logistic Regression with multiple penalty configurations."""
    
    print(f"\n{'='*60}")
    print("Training Logistic Regression models...")
    print(f"{'='*60}")
    
    timer.lap_start(f"LR Training ({feature_name})")
    
    # Define configurations
    configs = [
        {'name': 'no_penalty', 'penalty': None, 'solver': 'saga'},
        {'name': 'l1_lasso', 'penalty': 'l1', 'solver': 'saga', 'C_values': LR_C_VALUES},
        {'name': 'l2_ridge', 'penalty': 'l2', 'solver': 'saga', 'C_values': LR_C_VALUES},
        {'name': 'elasticnet', 'penalty': 'elasticnet', 'solver': 'saga', 
         'C_values': LR_C_VALUES, 'l1_ratio': [0.5]}
    ]
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    all_results = {}
    
    for cfg in configs:
        print(f"\n--- {cfg['name'].upper()} ---")
        
        # Build pipeline
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
            # Note: param names are prefixed with 'clf__' for pipeline
            param_grid = {'clf__C': cfg['C_values']}
            if cfg['penalty'] == 'elasticnet':
                param_grid['clf__l1_ratio'] = cfg.get('l1_ratio', [0.5])
        
        # GridSearchCV
        if param_grid:
            grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=CV_SCORING,
                                return_train_score=True, n_jobs=-1, verbose=1)
        else:
            grid = pipe
        
        start = time.time()
        if param_grid:
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            
            # Build detailed results for ALL models
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
                # Add individual fold scores
                for fold in range(CV_FOLDS):
                    model_result[f'split{fold}_test_score'] = float(grid.cv_results_[f'split{fold}_test_score'][i])
                all_models_results.append(model_result)
            
            # Sort by rank
            all_models_results.sort(key=lambda x: x['rank'])
            
            cv_results = {
                'best_score': grid.best_score_,
                'best_params': grid.best_params_,
                'all_models': all_models_results,
            }
            print(f"  Best params: {grid.best_params_}")
            print(f"  Best CV score: {grid.best_score_:.4f}")
        else:
            grid.fit(X_train, y_train)
            best_model = grid
            cv_results = None
            all_models_results = None
        
        train_time = time.time() - start
        
        # Create output directory
        run_dir = base_dir / f"LogisticRegression_{cfg['name']}_{feature_name}_{dataset}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            'model': 'LogisticRegression',
            'penalty': cfg['name'],
            'feature': feature_name,
            'dataset': dataset,
            'cv_folds': CV_FOLDS,
            'scoring': CV_SCORING,
            'best_params': cv_results['best_params'] if cv_results else {},
            'created_at': datetime.now().isoformat(),
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
        
        if cv_results:
            (run_dir / "cv_results.json").write_text(json.dumps(cv_results, indent=2, default=str))
            # Save readable CSV with all models
            if all_models_results:
                save_all_models_csv(all_models_results, run_dir / "all_models_results.csv")
        
        # Evaluate on test set (pipeline handles scaling automatically)
        title = f"LR ({cfg['name']}) - {feature_name} - {dataset}"
        metrics = evaluate_and_save(best_model, X_test, y_test, class_names,
                                    run_dir, title, cv_results, train_time)
        
        all_results[cfg['name']] = metrics
        # Add CV scores for summary
        if cv_results:
            all_results[cfg['name']]['cv_mean_test_score'] = cv_results['best_score']
            all_results[cfg['name']]['cv_std_test_score'] = cv_results['all_models'][0]['std_test_score'] if cv_results.get('all_models') else 0
            all_results[cfg['name']]['rank'] = 1  # Best model from grid search
        else:
            all_results[cfg['name']]['cv_mean_test_score'] = metrics['balanced_accuracy']
            all_results[cfg['name']]['cv_std_test_score'] = 0
            all_results[cfg['name']]['rank'] = '-'
        
        print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
        print(f"  Saved to: {run_dir}")
    
    timer.lap_end(f"LR Training ({feature_name})")
    
    return all_results

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        class_names: List[str], feature_name: str,
                        dataset: str, base_dir: Path):
    """Train Random Forest with GridSearchCV."""
    
    print(f"\n{'='*60}")
    print("Training Random Forest...")
    print(f"{'='*60}")
    print(f"Parameter grid: {RF_PARAM_GRID}")
    
    timer.lap_start(f"RF Training ({feature_name})")
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid = GridSearchCV(base_model, RF_PARAM_GRID, cv=cv, scoring=CV_SCORING,
                        return_train_score=True, n_jobs=-1, verbose=2)
    
    start = time.time()
    grid.fit(X_train, y_train)
    total_train_time = time.time() - start
    
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")
    
    # Build detailed results for ALL models
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
        # Add individual fold scores
        for fold in range(CV_FOLDS):
            model_result[f'split{fold}_test_score'] = float(grid.cv_results_[f'split{fold}_test_score'][i])
        all_models_results.append(model_result)
    
    # Sort by rank
    all_models_results.sort(key=lambda x: x['rank'])
    
    cv_results = {
        'best_score': grid.best_score_,
        'best_params': grid.best_params_,
        'all_models': all_models_results,
    }
    
    # Create output directory
    run_dir = base_dir / f"RandomForest_{feature_name}_{dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
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
    (run_dir / "cv_results.json").write_text(json.dumps(cv_results, indent=2, default=str))
    
    # Save readable CSV with all models
    save_all_models_csv(all_models_results, run_dir / "all_models_results.csv")
    
    # Evaluate BEST model on test set
    title = f"Random Forest (BEST) - {feature_name} - {dataset}"
    best_metrics = evaluate_and_save(grid.best_estimator_, X_test, y_test, class_names,
                                run_dir, title, cv_results, total_train_time)
    
    timer.lap_end(f"RF Training ({feature_name})")
    
    print(f"Best Model Accuracy: {best_metrics['overall_accuracy']:.4f}")
    print(f"Best Model Balanced Acc: {best_metrics['balanced_accuracy']:.4f}")
    print(f"Saved to: {run_dir}")
    
    # Return results for ALL hyperparameter combinations for summary
    all_results = {}
    for model_result in all_models_results:
        params = model_result['params']
        # Create readable param string
        param_str = "_".join([f"{k[:3]}{v}" for k, v in params.items()])
        model_name = f"n{params.get('n_estimators', '')}_d{params.get('max_depth', 'None')}_s{params.get('min_samples_split', '')}"
        
        all_results[model_name] = {
            'overall_accuracy': model_result['mean_test_score'],  # CV score as proxy
            'balanced_accuracy': model_result['mean_test_score'],
            'f1_macro': model_result['mean_test_score'],  # Using CV score
            'train_time_seconds': model_result['mean_fit_time'] * CV_FOLDS,
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
    """Print summary comparison of all models."""
    print(f"\n{'='*100}")
    print(f"SUMMARY - {dataset.upper()}")
    print(f"{'='*100}")
    print(f"{'Model':<55} {'CV_Score':>10} {'Std':>10} {'Rank':>6} {'Train(s)':>10}")
    print("-" * 100)
    
    # Separate LR and RF results for cleaner output
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
        # Sort RF by rank if available
        rf_sorted = sorted(rf_results.items(), key=lambda x: x[1].get('rank', 999))
        for name, metrics in rf_sorted:
            cv_score = metrics.get('cv_mean_test_score', metrics.get('balanced_accuracy', 0))
            std = metrics.get('cv_std_test_score', 0)
            rank = metrics.get('rank', '-')
            print(f"{name:<55} {cv_score:>10.4f} {std:>10.4f} {str(rank):>6} "
                  f"{metrics['train_time_seconds']:>10.1f}")

def main():
    set_seed(42)
    
    # Start global timer
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
        print("Invalid choice. Please enter 1, 2, or 3.")
    
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
        print("Invalid choice. Please enter 1, 2, or 3.")
    
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
        
        # Support both single value and list of bins
        bins_list = COLOR_HIST_BINS if isinstance(COLOR_HIST_BINS, list) else [COLOR_HIST_BINS]
        
        for bins in bins_list:
            print(f"\n>>> Processing Color Histogram with bins={bins} ({bins}x{bins}x{bins}={bins**3} features)")
            
            X_train_hist = extract_color_histogram_3d(Xtr, bins=bins)
            X_test_hist = extract_color_histogram_3d(Xte, bins=bins)
            
            feature_name = f"ColorHist_bins{bins}"
            
            if model_choice in [1, 3]:  # Logistic Regression
                lr_results = train_logistic_regression(
                    X_train_hist, ytr, X_test_hist, yte, class_names,
                    feature_name, dataset, base_dir
                )
                for name, metrics in lr_results.items():
                    all_results[f"LR_{name}_{feature_name}"] = metrics
            
            if model_choice in [2, 3]:  # Random Forest
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
        
        X_train_bovw, X_test_bovw = extract_sift_bovw(
            Xtr, Xte, n_clusters=BOVW_N_CLUSTERS
        )
        
        if model_choice in [1, 3]:  # Logistic Regression
            lr_results = train_logistic_regression(
                X_train_bovw, ytr, X_test_bovw, yte, class_names,
                "SIFT_BoVW", dataset, base_dir
            )
            for name, metrics in lr_results.items():
                all_results[f"LR_{name}_SIFT_BoVW"] = metrics
        
        if model_choice in [2, 3]:  # Random Forest
            rf_results = train_random_forest(
                X_train_bovw, ytr, X_test_bovw, yte, class_names,
                "SIFT_BoVW", dataset, base_dir
            )
            for name, metrics in rf_results.items():
                all_results[f"RF_{name}_SIFT_BoVW"] = metrics
    
    # Print summary
    print_summary(all_results, dataset)
    
    # Save overall summary including timing
    timer_data = timer.to_dict()
    summary_data = {
        'results': all_results,
        'timing': timer_data
    }
    summary_path = base_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2))
    
    # Print timer summary
    timer.summary()
    timer.stop()
    
    print(f"\nAll results saved to: {base_dir}")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()