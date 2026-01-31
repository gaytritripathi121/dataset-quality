import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats
from utils.logger import default_logger as logger


class DataProfiler:
    """Performs comprehensive data profiling and statistical analysis"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.profile = {}
    
    def profile_tabular(self, df: pd.DataFrame) -> Dict:
        """Profile tabular dataset"""
        logger.info("Profiling tabular dataset...")
        
        profile = {
            'basic_info': self._get_basic_info(df),
            'missing_values': self._analyze_missing_values(df),
            'duplicates': self._analyze_duplicates(df),
            'data_types': self._analyze_data_types(df),
            'statistics': self._get_statistics(df),
            'correlations': self._analyze_correlations(df)
        }
        
        return profile
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict:
        """Get basic dataset information"""
        return {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'columns': df.columns.tolist()
        }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values pattern"""
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df) * 100).round(2)
        
        return {
            'total_missing': int(df.isnull().sum().sum()),
            'missing_percentage': round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
            'columns_with_missing': {
                col: {'count': int(count), 'percentage': float(pct)}
                for col, count, pct in zip(df.columns, missing_count, missing_pct)
                if count > 0
            }
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict:
        """Analyze duplicate records"""
        n_duplicates = df.duplicated().sum()
        
        return {
            'n_duplicates': int(n_duplicates),
            'duplicate_percentage': round(n_duplicates / len(df) * 100, 2),
            'duplicate_indices': df[df.duplicated()].index.tolist()[:100]  # First 100
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict:
        """Analyze data types distribution"""
        return {
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
        }
    
    def _get_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistical summaries"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {}
        
        stats_dict = {}
        for col in numeric_df.columns:
            stats_dict[col] = {
                'mean': float(numeric_df[col].mean()),
                'std': float(numeric_df[col].std()),
                'min': float(numeric_df[col].min()),
                'max': float(numeric_df[col].max()),
                'median': float(numeric_df[col].median()),
                'skewness': float(numeric_df[col].skew()),
                'kurtosis': float(numeric_df[col].kurtosis())
            }
        
        return stats_dict
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze feature correlations"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {}
        
        corr_matrix = numeric_df.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_matrix.iloc[i, j])
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr_pairs
        }