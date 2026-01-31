import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_loader import DatasetLoader
from src.data_profiler import DataProfiler
from src.quality_checker import QualityChecker
from src.scoring_engine import ScoringEngine
from utils.helpers import create_sample_data


class TestDatasetLoader:
    """Test DatasetLoader functionality"""
    
    def test_load_csv(self, tmp_path):
        """Test loading CSV file"""
        # Create sample CSV
        df = create_sample_data(100, 5)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Load
        loader = DatasetLoader()
        data, dtype, metadata = loader.load(csv_path)
        
        assert dtype == 'tabular'
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
    
    def test_detect_type(self):
        """Test type detection"""
        loader = DatasetLoader()
        
        assert loader._detect_file_type(Path("test.csv")) == 'tabular'
        assert loader._detect_file_type(Path("test.jpg")) == 'image'
        assert loader._detect_file_type(Path("test.txt")) == 'text'


class TestDataProfiler:
    """Test DataProfiler functionality"""
    
    def test_profile_tabular(self):
        """Test tabular profiling"""
        df = create_sample_data(100, 5)
        
        profiler = DataProfiler()
        profile = profiler.profile_tabular(df)
        
        assert 'basic_info' in profile
        assert 'missing_values' in profile
        assert 'duplicates' in profile
        assert profile['basic_info']['n_rows'] == 100


class TestQualityChecker:
    """Test QualityChecker functionality"""
    
    def test_missing_values_check(self):
        """Test missing values detection"""
        df = create_sample_data(100, 5)
        
        profiler = DataProfiler()
        profile = profiler.profile_tabular(df)
        
        checker = QualityChecker()
        checks = checker.check_tabular(df, profile)
        
        assert 'missing_values_check' in checks
        assert 'status' in checks['missing_values_check']


class TestScoringEngine:
    """Test ScoringEngine functionality"""
    
    def test_score_calculation(self):
        """Test score calculation"""
        df = create_sample_data(100, 5)
        
        profiler = DataProfiler()
        profile = profiler.profile_tabular(df)
        
        checker = QualityChecker()
        checks = checker.check_tabular(df, profile)
        
        scorer = ScoringEngine()
        scores = scorer.calculate_score(profile, checks, {})
        
        assert 'overall_score' in scores
        assert 0 <= scores['overall_score'] <= 100
        assert 'rating' in scores


def test_sample_data_creation():
    """Test sample data generation"""
    df = create_sample_data(100, 5, 0.1)
    
    assert len(df) == 100
    assert len(df.columns) == 7  # 5 features + category + target
    assert df.isnull().any().any()  # Should have some missing values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
