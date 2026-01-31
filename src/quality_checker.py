from typing import Dict, Optional
import logging

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class QualityChecker:
    """Performs rule-based data quality checks"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.thresholds = self.config.get("quality_thresholds", {})

    def check_tabular(self, df: pd.DataFrame, profile: Dict) -> Dict:
        """Perform quality checks on tabular data"""
        logger.info("Performing quality checks...")

        checks = {
            "missing_values_check": self._check_missing_values(profile),
            "duplicates_check": self._check_duplicates(profile),
            "class_balance_check": self._check_class_balance(df),
            "outliers_check": self._check_outliers(df),
            "data_types_check": self._check_data_types(df),
        }

        return checks

    def _check_missing_values(self, profile: Dict) -> Dict:
        missing_pct = profile["missing_values"]["missing_percentage"]
        thresholds = self.thresholds.get("missing_values", {})

        if missing_pct < thresholds.get("excellent", 2):
            status = "excellent"
        elif missing_pct < thresholds.get("good", 5):
            status = "good"
        elif missing_pct < thresholds.get("fair", 15):
            status = "fair"
        else:
            status = "poor"

        return {
            "status": status,
            "missing_percentage": missing_pct,
            "severity": "high" if status == "poor" else "medium" if status == "fair" else "low",
        }

    def _check_duplicates(self, profile: Dict) -> Dict:
        dup_pct = profile["duplicates"]["duplicate_percentage"]
        thresholds = self.thresholds.get("duplicates", {})

        if dup_pct < thresholds.get("excellent", 1):
            status = "excellent"
        elif dup_pct < thresholds.get("good", 3):
            status = "good"
        elif dup_pct < thresholds.get("fair", 10):
            status = "fair"
        else:
            status = "poor"

        return {
            "status": status,
            "duplicate_percentage": dup_pct,
            "severity": "high" if status == "poor" else "medium" if status == "fair" else "low",
        }

    def _check_class_balance(self, df: pd.DataFrame) -> Dict:
        potential_targets = ["target", "label", "class", "y"]
        target_col = next((c for c in potential_targets if c in df.columns), None)

        if target_col is None and len(df.columns) > 0:
            target_col = df.columns[-1]

        if target_col and df[target_col].nunique() < 20:
            value_counts = df[target_col].value_counts()
            imbalance_ratio = (
                value_counts.max() / value_counts.min()
                if value_counts.min() > 0
                else float("inf")
            )

            thresholds = self.thresholds.get("class_imbalance", {})
            if imbalance_ratio < thresholds.get("excellent", 1.5):
                status = "excellent"
            elif imbalance_ratio < thresholds.get("good", 3.0):
                status = "good"
            elif imbalance_ratio < thresholds.get("fair", 5.0):
                status = "fair"
            else:
                status = "poor"

            return {
                "status": status,
                "imbalance_ratio": float(imbalance_ratio),
                "class_distribution": value_counts.to_dict(),
                "severity": "high" if status == "poor" else "medium" if status == "fair" else "low",
            }

        return {"status": "not_applicable", "severity": "low"}

    def _check_outliers(self, df: pd.DataFrame) -> Dict:
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {"status": "not_applicable", "severity": "low"}

        outlier_counts = {}
        total_outliers = 0

        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()
            outlier_counts[col] = int(outliers)
            total_outliers += outliers

        outlier_pct = (total_outliers / (len(df) * len(numeric_df.columns))) * 100

        if outlier_pct < 2:
            status = "excellent"
        elif outlier_pct < 5:
            status = "good"
        elif outlier_pct < 10:
            status = "fair"
        else:
            status = "poor"

        return {
            "status": status,
            "outlier_percentage": round(outlier_pct, 2),
            "outliers_per_column": outlier_counts,
            "severity": "medium" if status in ["fair", "poor"] else "low",
        }

    def _check_data_types(self, df: pd.DataFrame) -> Dict:
        issues = []

        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    pd.to_numeric(df[col].dropna())
                    issues.append(f"{col}: numeric values stored as strings")
                except Exception:
                    pass

        return {
            "status": "good" if not issues else "fair",
            "issues": issues,
            "severity": "low",
        }
