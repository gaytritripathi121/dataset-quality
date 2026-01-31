"""
Logging configuration for Dataset Quality Auditor
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


class AuditorLogger:
    """
    Custom logger for the Dataset Quality Auditor
    """
    
    def __init__(self, log_file: Optional[str] = None, level: str = "INFO"):
        """
        Initialize logger
        
        Args:
            log_file: Path to log file (optional)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.level = level
        self.log_file = log_file
        
        # Remove default logger
        logger.remove()
        
        # Add console handler with custom format
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True
        )
        
        # Add file handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
                level=level,
                rotation="10 MB",  # Rotate after 10 MB
                retention="30 days",  # Keep logs for 30 days
                compression="zip"  # Compress rotated logs
            )
    
    @staticmethod
    def get_logger():
        """Get logger instance"""
        return logger


def setup_logger(config: dict = None) -> logger:
    """
    Setup logger with configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger
    """
    if config is None:
        config = {
            'level': 'INFO',
            'log_to_file': False,
            'log_file_path': 'outputs/logs/audit.log'
        }
    
    level = config.get('level', 'INFO')
    log_to_file = config.get('log_to_file', False)
    log_file_path = config.get('log_file_path', 'outputs/logs/audit.log') if log_to_file else None
    
    auditor_logger = AuditorLogger(log_file=log_file_path, level=level)
    return auditor_logger.get_logger()


# Create default logger instance
default_logger = setup_logger()