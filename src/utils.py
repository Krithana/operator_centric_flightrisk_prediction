"""
Utility Functions
Helper functions for the aviation safety prediction system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import FIGURES_DIR, MODELS_DIR


def print_section_header(title: str, char: str = "=", width: int = 80):
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        char: Character to use for border
        width: Width of the header
    """
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def print_metrics(metrics: dict, title: str = "Model Performance"):
    """
    Print model performance metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Title for the metrics table
    """
    print(f"\n{title}")
    print("-" * 60)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:<20}: {value:.4f}")
        else:
            print(f"{metric_name:<20}: {value}")
    print("-" * 60)


def save_results(results: dict, filename: str, directory: Path = None):
    """
    Save results dictionary to pickle file.
    
    Args:
        results: Dictionary of results to save
        filename: Output filename
        directory: Directory to save to (default: MODELS_DIR)
    """
    if directory is None:
        directory = MODELS_DIR
    
    filepath = directory / filename
    joblib.dump(results, filepath)
    print(f"💾 Saved results to {filepath}")


def setup_plotting_style():
    """Setup consistent plotting style for research papers."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Academic paper style settings
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'


def save_figure(filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """
    Save current matplotlib figure with research paper quality.
    
    Args:
        filename: Output filename
        dpi: Resolution (dots per inch)
        bbox_inches: Bounding box setting
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
    print(f"📊 Saved figure: {filepath.name}")


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a float as a percentage string.
    
    Args:
        value: Float value between 0 and 1
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def create_results_summary(metrics_dict: dict) -> pd.DataFrame:
    """
    Create a summary DataFrame from metrics dictionary.
    
    Args:
        metrics_dict: Dictionary of metrics for multiple models
    
    Returns:
        DataFrame with formatted results
    """
    df = pd.DataFrame(metrics_dict).T
    
    # Format percentage columns
    percentage_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    return df


def print_confusion_matrix_stats(cm: np.ndarray, labels: list = None):
    """
    Print detailed statistics from confusion matrix.
    
    Args:
        cm: Confusion matrix (2x2 numpy array)
        labels: Class labels
    """
    if labels is None:
        labels = ['Non-Severe', 'Severe']
    
    tn, fp, fn, tp = cm.ravel()
    
    print("\n📊 Confusion Matrix Statistics:")
    print("=" * 60)
    print(f"True Negatives (TN):  {tn:,} - Correctly identified {labels[0]}")
    print(f"False Positives (FP): {fp:,} - {labels[0]} incorrectly flagged as {labels[1]}")
    print(f"False Negatives (FN): {fn:,} - {labels[1]} incorrectly classified as {labels[0]}")
    print(f"True Positives (TP):  {tp:,} - Correctly identified {labels[1]}")
    print("=" * 60)
    
    total = tn + fp + fn + tp
    print(f"\nTotal predictions: {total:,}")
    print(f"Correct predictions: {tn + tp:,} ({(tn+tp)/total*100:.2f}%)")
    print(f"Incorrect predictions: {fp + fn:,} ({(fp+fn)/total*100:.2f}%)")
    
    # Additional metrics
    if tp + fn > 0:
        detection_rate = tp / (tp + fn)
        print(f"\nDetection Rate (Recall): {detection_rate*100:.2f}%")
        print(f"Missed {labels[1]} cases: {fn:,} ({fn/(tp+fn)*100:.1f}%)")
    
    if tn + fp > 0:
        specificity = tn / (tn + fp)
        false_alarm_rate = fp / (tn + fp)
        print(f"Specificity: {specificity*100:.2f}%")
        print(f"False Alarm Rate: {false_alarm_rate*100:.2f}%")


def calculate_improvement(baseline_metrics: dict, improved_metrics: dict, metric_name: str = 'Accuracy') -> dict:
    """
    Calculate improvement between baseline and improved models.
    
    Args:
        baseline_metrics: Metrics dictionary for baseline model
        improved_metrics: Metrics dictionary for improved model
        metric_name: Metric to calculate improvement for
    
    Returns:
        Dictionary with improvement statistics
    """
    baseline_value = baseline_metrics.get(metric_name, 0)
    improved_value = improved_metrics.get(metric_name, 0)
    
    absolute_improvement = improved_value - baseline_value
    relative_improvement = (absolute_improvement / baseline_value * 100) if baseline_value > 0 else 0
    
    return {
        'baseline': baseline_value,
        'improved': improved_value,
        'absolute_improvement': absolute_improvement,
        'relative_improvement_pct': relative_improvement,
        'metric': metric_name
    }


def generate_timestamp_str() -> str:
    """
    Generate a timestamp string for file naming.
    
    Returns:
        Formatted timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_model(model_path: Path):
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Loaded model object
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"📂 Loaded model from {model_path.name}")
    return model


def save_model(model, filename: str, directory: Path = None, metadata: dict = None):
    """
    Save a trained model to disk with optional metadata.
    
    Args:
        model: Trained model object
        filename: Output filename
        directory: Directory to save to (default: MODELS_DIR)
        metadata: Optional metadata dictionary to save
    """
    if directory is None:
        directory = MODELS_DIR
    
    directory.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = directory / filename
    joblib.dump(model, model_path)
    print(f"💾 Saved model to {model_path.name}")
    
    # Save metadata if provided
    if metadata is not None:
        metadata_path = directory / f"{Path(filename).stem}_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        print(f"💾 Saved metadata to {metadata_path.name}")


def compare_models(results_list: list[dict]) -> pd.DataFrame:
    """
    Create a comparison table of multiple models.
    
    Args:
        results_list: List of dictionaries containing model results
    
    Returns:
        DataFrame with model comparison
    """
    df = pd.DataFrame(results_list)
    
    # Sort by accuracy
    if 'Accuracy' in df.columns:
        df = df.sort_values('Accuracy', ascending=False)
    
    return df


def print_paper_ready_table(df: pd.DataFrame, title: str = "Results"):
    """
    Print a DataFrame in a format suitable for copying to a paper.
    
    Args:
        df: DataFrame to print
        title: Table title
    """
    print(f"\n{title}")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


# Risk classification thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.5,
    'high': 0.7
}


def classify_risk(probability: float) -> str:
    """
    Classify risk level based on predicted probability.
    
    Args:
        probability: Predicted probability of severe accident
    
    Returns:
        Risk level string ('Low', 'Medium', 'High', 'Very High')
    """
    if probability < RISK_THRESHOLDS['low']:
        return 'Low'
    elif probability < RISK_THRESHOLDS['medium']:
        return 'Medium'
    elif probability < RISK_THRESHOLDS['high']:
        return 'High'
    else:
        return 'Very High'


if __name__ == "__main__":
    # Test utilities
    print_section_header("Testing Utilities")
    
    test_metrics = {
        'Accuracy': 0.7683,
        'Precision': 0.5485,
        'Recall': 0.4368,
        'F1-Score': 0.5050,
        'ROC-AUC': 0.7772
    }
    
    print_metrics(test_metrics, "Test Model Metrics")
    
    print("\nFormatted percentage:", format_percentage(0.7683))
    print("Risk classification:", classify_risk(0.85))
    print("Timestamp:", generate_timestamp_str())
    
    print("\n✅ All utilities working correctly!")
