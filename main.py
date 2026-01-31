import sys
import warnings
from pathlib import Path
from typing import Dict, Any

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset_loader import DatasetLoader
from src.data_profiler import DataProfiler
from src.quality_checker import QualityChecker
from src.ml_anomaly_detector import MLAnomalyDetector
from src.scoring_engine import ScoringEngine
from src.visualizer import Visualizer
from src.report_generator import ReportGenerator
from utils.helpers import load_config, ensure_dir, get_timestamp
from utils.logger import setup_logger


class DatasetQualityAuditor:
    """Orchestrates the dataset quality audit pipeline"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config.get("logging", {}))

        self.loader = DatasetLoader(self.config)
        self.profiler = DataProfiler(self.config)
        self.quality_checker = QualityChecker(self.config)
        self.anomaly_detector = MLAnomalyDetector(self.config)
        self.scoring_engine = ScoringEngine(self.config)
        self.visualizer = Visualizer(self.config)
        self.report_generator = ReportGenerator(self.config)

        self.logger.info("Dataset Quality Auditor initialized")

    def audit_dataset(self, file_path: str, output_dir: str = None) -> Dict[str, Any]:
        self.logger.info("=" * 70)
        self.logger.info("Starting Dataset Quality Audit")
        self.logger.info("=" * 70)

        if output_dir is None:
            output_dir = f"outputs/audit_{get_timestamp()}"

        output_path = ensure_dir(output_dir)
        self.logger.info(f"Output directory: {output_path}")

        try:
            self.logger.info("\n[Step 1/6] Loading Dataset...")
            data, dataset_type, metadata = self.loader.load(file_path)
            self.logger.info(
                f"✓ Loaded {metadata['dataset_type']} dataset with {metadata['n_samples']} samples"
            )

            self.logger.info("\n[Step 2/6] Profiling Data...")
            if dataset_type == "tabular":
                profile = self.profiler.profile_tabular(data)
                self.logger.info("✓ Profiling complete")
            else:
                profile = {}
                self.logger.info("✓ Profiling skipped for non-tabular data")

            self.logger.info("\n[Step 3/6] Running Quality Checks...")
            if dataset_type == "tabular":
                quality_checks = self.quality_checker.check_tabular(data, profile)
                self.logger.info(f"✓ Completed {len(quality_checks)} quality checks")
            else:
                quality_checks = {}
                self.logger.info("✓ Quality checks skipped for non-tabular data")

            self.logger.info("\n[Step 4/6] Running ML Anomaly Detection...")
            if dataset_type == "tabular":
                anomaly_results = self.anomaly_detector.detect_tabular_anomalies(data)
                self.logger.info("✓ Anomaly detection complete")
            else:
                anomaly_results = {}
                self.logger.info("✓ Anomaly detection skipped for non-tabular data")

            self.logger.info("\n[Step 5/6] Calculating Health Score...")
            scores = self.scoring_engine.calculate_score(
                profile, quality_checks, anomaly_results
            )
            self.logger.info(
                f"✓ Dataset Health Score: {scores['overall_score']}/100 ({scores['rating']})"
            )

            self.logger.info("\n[Step 6/6] Generating Reports...")

            plot_paths = self.visualizer.generate_all_plots(
                output_dir=output_path / "visualizations",
                df=data if dataset_type == "tabular" else None,
                profile=profile,
                quality_checks=quality_checks,
                anomaly_results=anomaly_results,
                scores=scores,
            )

            report_paths = self.report_generator.generate_report(
                output_dir=output_path / "reports",
                metadata=metadata,
                profile=profile,
                quality_checks=quality_checks,
                anomaly_results=anomaly_results,
                scores=scores,
                plot_paths=plot_paths,
            )

            self.logger.info("✓ Reports generated")

            self._print_summary(metadata, scores, quality_checks, anomaly_results)

            self.logger.info("\n" + "=" * 70)
            self.logger.info("Audit Complete!")
            self.logger.info("=" * 70)

            return {
                "metadata": metadata,
                "profile": profile,
                "quality_checks": quality_checks,
                "anomaly_results": anomaly_results,
                "scores": scores,
                "output_dir": str(output_path),
                "report_paths": report_paths,
                "plot_paths": plot_paths,
            }

        except Exception as e:
            self.logger.error(f"Audit failed: {str(e)}")
            raise

    def _print_summary(
        self,
        metadata: Dict,
        scores: Dict,
        quality_checks: Dict,
        anomaly_results: Dict,
    ):
        """Print audit summary to console"""

        print("\n" + "=" * 70)
        print("DATASET QUALITY AUDIT SUMMARY")
        print("=" * 70)

        print(f"\nDataset: {metadata['file_name']}")
        print(f"Type: {metadata['dataset_type']}")
        print(f"Samples: {metadata['n_samples']}")

        if "n_features" in metadata:
            print(f"Features: {metadata['n_features']}")

        print("\n" + "-" * 70)
        print("QUALITY METRICS")
        print("-" * 70)

        print(
            f"\nOverall Health Score: {scores['overall_score']}/100 ({scores['rating']})"
        )
        print(f"Recommendation: {scores['recommendation']}")

        print("\nComponent Scores (Weighted):")
        weighted_scores = {
            comp: round(score * scores["weights"].get(comp, 0.25), 2)
            for comp, score in scores["component_scores"].items()
        }

        for component, weighted_score in weighted_scores.items():
            percentage = scores["component_scores"][component]
            print(
                f"  ├─ {component.capitalize()}: {weighted_score}/25 ({percentage:.0f}%)"
            )

        print("\nScore Calculation:")
        total = sum(weighted_scores.values())
        calc_parts = " + ".join(f"{v:.2f}" for v in weighted_scores.values())
        print(f"  {calc_parts} = {total:.2f}/100")

        print("\n" + "-" * 70)
        print("KEY FINDINGS")
        print("-" * 70)

        if "missing_values_check" in quality_checks:
            missing = quality_checks["missing_values_check"]
            print(
                f"\n• Missing Values: {missing.get('missing_percentage', 0):.2f}% "
                f"[{missing.get('status', 'unknown').upper()}]"
            )

        if "duplicates_check" in quality_checks:
            dup = quality_checks["duplicates_check"]
            print(
                f"• Duplicates: {dup.get('duplicate_percentage', 0):.2f}% "
                f"[{dup.get('status', 'unknown').upper()}]"
            )

        if "class_balance_check" in quality_checks:
            balance = quality_checks["class_balance_check"]
            if balance.get("status") != "not_applicable":
                ratio = balance.get("imbalance_ratio", 0)
                print(
                    f"• Class Imbalance: {ratio:.2f}:1 "
                    f"[{balance.get('status', 'unknown').upper()}]"
                )
                if ratio > 5:
                    print(
                        "  └─ Common in churn/attrition datasets; manageable with standard techniques"
                    )

        if "combined" in anomaly_results:
            anomaly = anomaly_results["combined"]
            print(
                f"\n• Statistical Outliers: {anomaly['n_anomalies']} detected "
                f"({anomaly['anomaly_percentage']:.2f}%)"
            )
            print("  └─ ML-based detection; not necessarily data errors")
            print("  └─ Includes rare combinations, edge cases, legitimate extremes")
            if anomaly["anomaly_percentage"] < 15:
                print("  └─ Rate is within acceptable range for real-world data")

        print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Quality Auditor")
    parser.add_argument("--file", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Config file path"
    )

    args = parser.parse_args()

    auditor = DatasetQualityAuditor(config_path=args.config)
    results = auditor.audit_dataset(
        file_path=args.file,
        output_dir=args.output,
    )

    print(f"\n✓ Results saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()
