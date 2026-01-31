"""
Utility helper functions for the Dataset Quality Auditor
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from datetime import datetime


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """
    Get current timestamp as formatted string
    
    Returns:
        Timestamp string (YYYY-MM-DD_HH-MM-SS)
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """
    Save dictionary as JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load JSON file as dictionary
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """
    Get file size in megabytes
    
    Args:
        filepath: File path
        
    Returns:
        File size in MB
    """
    return os.path.getsize(filepath) / (1024 * 1024)


def detect_file_type(filepath: Union[str, Path]) -> str:
    """
    Detect file type from extension
    
    Args:
        filepath: File path
        
    Returns:
        File type ('csv', 'excel', 'parquet', 'image', 'text')
    """
    extension = Path(filepath).suffix.lower()
    
    extension_map = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.parquet': 'parquet',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.png': 'image',
        '.bmp': 'image',
        '.txt': 'text',
        '.json': 'json'
    }
    
    return extension_map.get(extension, 'unknown')


def calculate_percentage(part: float, whole: float) -> float:
    """
    Calculate percentage safely
    
    Args:
        part: Partial value
        whole: Total value
        
    Returns:
        Percentage (0-100)
    """
    if whole == 0:
        return 0.0
    return round((part / whole) * 100, 2)


def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get memory usage statistics for DataFrame
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with memory statistics
    """
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    return {
        'total_mb': round(memory_mb, 2),
        'per_column': {
            col: round(df[col].memory_usage(deep=True) / (1024 * 1024), 2)
            for col in df.columns
        }
    }


def normalize_scores(scores: np.ndarray, min_val: float = 0, max_val: float = 100) -> np.ndarray:
    """
    Normalize scores to specified range
    
    Args:
        scores: Array of scores
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
        
    Returns:
        Normalized scores
    """
    if len(scores) == 0:
        return scores
    
    score_min = np.min(scores)
    score_max = np.max(scores)
    
    if score_max == score_min:
        return np.full_like(scores, (min_val + max_val) / 2)
    
    normalized = (scores - score_min) / (score_max - score_min)
    return normalized * (max_val - min_val) + min_val


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_categorical_columns(df: pd.DataFrame, threshold: int = 10) -> List[str]:
    """
    Identify categorical columns in DataFrame
    
    Args:
        df: Input DataFrame
        threshold: Max unique values to consider categorical
        
    Returns:
        List of categorical column names
    """
    categorical_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() <= threshold:
            categorical_cols.append(col)
    
    return categorical_cols


def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify numerical columns in DataFrame
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of numerical column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def remove_outliers_iqr(data: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """
    Remove outliers using IQR method
    
    Args:
        data: Input array
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        Boolean mask (True for non-outliers)
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (data >= lower_bound) & (data <= upper_bound)


def calculate_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        labels: Array of class labels
        
    Returns:
        Dictionary mapping class to weight
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    weights = {
        int(cls): total / (len(unique) * count)
        for cls, count in zip(unique, counts)
    }
    
    return weights


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def create_sample_data(n_samples: int = 1000, n_features: int = 10, 
                       anomaly_ratio: float = 0.1) -> pd.DataFrame:
    """
    Create sample tabular data for testing
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        anomaly_ratio: Ratio of anomalies to inject
        
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    # Generate normal data
    data = np.random.randn(n_samples, n_features)
    
    # Add some anomalies
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    data[anomaly_indices] += np.random.randn(n_anomalies, n_features) * 5
    
    # Create DataFrame
    df = pd.DataFrame(
        data,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add categorical column
    df['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
    
    # Add target column
    df['target'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Inject some missing values
    missing_mask = np.random.random(df.shape) < 0.05
    df = df.mask(missing_mask)
    
    return df


def print_section_header(title: str, width: int = 60) -> None:
    """
    Print formatted section header
    
    Args:
        title: Section title
        width: Width of header
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def format_metric_table(metrics: Dict[str, Any]) -> str:
    """
    Format metrics as ASCII table
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted table string
    """
    lines = []
    max_key_len = max(len(str(k)) for k in metrics.keys())
    
    for key, value in metrics.items():
        if isinstance(value, float):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        
        lines.append(f"{key:<{max_key_len}} : {value_str}")
    
    return "\n".join(lines)


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform basic validation on DataFrame
    
    Args:
        df: Input DataFrame
        
    Returns:
        Validation results dictionary
    """
    return {
        'is_valid': True if not df.empty else False,
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'has_duplicates': df.duplicated().any(),
        'has_missing': df.isnull().any().any(),
        'memory_mb': round(df.memory_usage(deep=True).sum() / (1024**2), 2)
    }