
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import json
from pathlib import Path
from typing import Any, Tuple, Dict, Union

import pandas as pd
import numpy as np

from utils.helpers import detect_file_type, get_file_size_mb, ensure_dir
from utils.logger import default_logger as logger


class DatasetLoader:
    """
    Handles loading datasets of various types (tabular, image, text)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize DatasetLoader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.dataset_type = None
        self.data = None
        self.metadata = {}
    
    def load(self, file_path: Union[str, Path]) -> Tuple[Any, str, Dict]:
        """
        Load dataset and automatically detect type
        
        Args:
            file_path: Path to dataset file or directory
            
        Returns:
            Tuple of (data, dataset_type, metadata)
        """
        file_path = Path(file_path)
        
        logger.info(f"Loading dataset from: {file_path}")
        
        # Check if path exists
        if not file_path.exists():
            raise FileNotFoundError(f"Path not found: {file_path}")
        
        # Detect dataset type
        if file_path.is_dir():
            self.dataset_type = self._detect_directory_type(file_path)
        else:
            self.dataset_type = self._detect_file_type(file_path)
        
        logger.info(f"Detected dataset type: {self.dataset_type}")
        
        # Load based on type
        if self.dataset_type == 'tabular':
            self.data = self._load_tabular(file_path)
        elif self.dataset_type == 'image':
            self.data = self._load_images(file_path)
        elif self.dataset_type == 'text':
            self.data = self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        # Generate metadata
        self.metadata = self._generate_metadata(file_path)
        
        logger.info(f"Successfully loaded {self.metadata.get('n_samples', 0)} samples")
        
        return self.data, self.dataset_type, self.metadata
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect dataset type from file"""
        extension = file_path.suffix.lower()
        
        if extension in ['.csv', '.xlsx', '.xls', '.parquet', '.json']:
            return 'tabular'
        elif extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            return 'image'
        elif extension in ['.txt']:
            return 'text'
        else:
            # Try to infer from content
            return self._infer_type_from_content(file_path)
    
    def _detect_directory_type(self, dir_path: Path) -> str:
        """Detect dataset type from directory contents"""
        files = list(dir_path.glob('*'))
        
        if not files:
            raise ValueError("Empty directory")
        
        # Check first few files
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        text_exts = {'.txt'}
        
        sample_files = files[:10]
        
        # Count file types
        image_count = sum(1 for f in sample_files if f.suffix.lower() in image_exts)
        text_count = sum(1 for f in sample_files if f.suffix.lower() in text_exts)
        
        if image_count > len(sample_files) * 0.5:
            return 'image'
        elif text_count > len(sample_files) * 0.5:
            return 'text'
        else:
            return 'tabular'
    
    def _infer_type_from_content(self, file_path: Path) -> str:
        """Infer dataset type by examining file content"""
        try:
            # Try reading as CSV
            pd.read_csv(file_path, nrows=5)
            return 'tabular'
        except:
            pass
        
        try:
            # Try opening as image
            Image.open(file_path)
            return 'image'
        except:
            pass
        
        # Default to text
        return 'text'
    
    def _load_tabular(self, file_path: Path) -> pd.DataFrame:
        """
        Load tabular dataset
        
        Args:
            file_path: Path to file
            
        Returns:
            DataFrame
        """
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.csv':
                df = pd.read_csv(file_path)
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif extension == '.parquet':
                df = pd.read_parquet(file_path)
            elif extension == '.json':
                df = pd.read_json(file_path)
            else:
                # Try CSV as fallback
                df = pd.read_csv(file_path)
            
            logger.info(f"Loaded tabular data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load tabular data: {str(e)}")
            raise
    
    def _load_images(self, path: Path) -> Dict:
        """
        Load image dataset
        
        Args:
            path: Path to image file or directory
            
        Returns:
            Dictionary with image information
        """
        image_data = {
            'file_paths': [],
            'images': [],
            'labels': [],
            'metadata': []
        }
        
        if path.is_file():
            # Single image
            image_paths = [path]
        else:
            # Directory of images
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_paths = [
                f for f in path.glob('**/*')
                if f.suffix.lower() in image_exts
            ]
        
        logger.info(f"Found {len(image_paths)} images")
        
        for img_path in image_paths[:1000]:  # Limit for efficiency
            try:
                img = Image.open(img_path)
                
                image_data['file_paths'].append(str(img_path))
                image_data['images'].append(np.array(img))
                
                # Extract label from parent directory name
                label = img_path.parent.name
                image_data['labels'].append(label)
                
                # Store metadata
                image_data['metadata'].append({
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format
                })
                
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {str(e)}")
        
        return image_data
    
    def _load_text(self, path: Path) -> Dict:
        """
        Load text dataset
        
        Args:
            path: Path to text file or directory
            
        Returns:
            Dictionary with text data
        """
        text_data = {
            'texts': [],
            'labels': [],
            'file_paths': []
        }
        
        if path.is_file():
            if path.suffix == '.csv':
                # CSV with text column
                df = pd.read_csv(path)
                # Find text column (usually the longest strings)
                text_col = None
                for col in df.columns:
                    if df[col].dtype == 'object':
                        avg_len = df[col].astype(str).str.len().mean()
                        if avg_len > 50:  # Heuristic for text column
                            text_col = col
                            break
                
                if text_col:
                    text_data['texts'] = df[text_col].tolist()
                    # Try to find label column
                    label_candidates = [c for c in df.columns if c != text_col]
                    if label_candidates:
                        text_data['labels'] = df[label_candidates[0]].tolist()
                else:
                    raise ValueError("No text column found in CSV")
            else:
                # Plain text file
                with open(path, 'r', encoding='utf-8') as f:
                    text_data['texts'] = [line.strip() for line in f if line.strip()]
        else:
            # Directory of text files
            text_files = list(path.glob('**/*.txt'))
            
            for txt_file in text_files[:1000]:  # Limit for efficiency
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            text_data['texts'].append(content)
                            text_data['labels'].append(txt_file.parent.name)
                            text_data['file_paths'].append(str(txt_file))
                except Exception as e:
                    logger.warning(f"Failed to load text file {txt_file}: {str(e)}")
        
        logger.info(f"Loaded {len(text_data['texts'])} text samples")
        return text_data
    
    def _generate_metadata(self, file_path: Path) -> Dict:
        """
        Generate metadata about the loaded dataset
        
        Args:
            file_path: Path to dataset
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'dataset_type': self.dataset_type,
            'file_size_mb': None,
            'n_samples': 0,
            'n_features': 0
        }
        
        # File size
        if file_path.is_file():
            metadata['file_size_mb'] = round(get_file_size_mb(file_path), 2)
        
        # Type-specific metadata
        if self.dataset_type == 'tabular' and isinstance(self.data, pd.DataFrame):
            metadata['n_samples'] = len(self.data)
            metadata['n_features'] = len(self.data.columns)
            metadata['columns'] = self.data.columns.tolist()
            metadata['dtypes'] = {col: str(dtype) for col, dtype in self.data.dtypes.items()}
            
        elif self.dataset_type == 'image' and isinstance(self.data, dict):
            metadata['n_samples'] = len(self.data['images'])
            metadata['n_classes'] = len(set(self.data['labels']))
            metadata['unique_labels'] = list(set(self.data['labels']))
            
        elif self.dataset_type == 'text' and isinstance(self.data, dict):
            metadata['n_samples'] = len(self.data['texts'])
            if self.data['labels']:
                metadata['n_classes'] = len(set(self.data['labels']))
                metadata['unique_labels'] = list(set(self.data['labels']))
        
        return metadata
    
    def get_sample(self, n: int = 5) -> Any:
        """
        Get sample from loaded dataset
        
        Args:
            n: Number of samples
            
        Returns:
            Sample data
        """
        if self.data is None:
            raise ValueError("No dataset loaded")
        
        if self.dataset_type == 'tabular':
            return self.data.head(n)
        elif self.dataset_type in ['image', 'text']:
            indices = min(n, len(self.data.get('images', self.data.get('texts', []))))
            sample = {}
            for key in self.data:
                if isinstance(self.data[key], list):
                    sample[key] = self.data[key][:indices]
            return sample
        
        return None