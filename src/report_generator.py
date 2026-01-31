import json
from datetime import datetime
from typing import Dict, List
from jinja2 import Template
from utils.helpers import ensure_dir
from utils.logger import default_logger as logger


class ReportGenerator:
    """Generates comprehensive quality audit reports"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def generate_report(
        self,
        output_dir: str,
        metadata: Dict,
        profile: Dict,
        quality_checks: Dict,
        anomaly_results: Dict,
        scores: Dict,
        plot_paths: List[str]
    ) -> Dict:
        """Generate comprehensive audit report"""
        logger.info("Generating audit report...")
        
        output_dir = ensure_dir(output_dir)
        
        report_data = {
            'metadata': metadata,
            'profile': profile,
            'quality_checks': quality_checks,
            'anomaly_results': anomaly_results,
            'scores': scores,
            'plot_paths': plot_paths,
            'timestamp': datetime.now().isoformat(),
            'recommendations': self._generate_recommendations(
                quality_checks, anomaly_results, scores
            )
        }
        
        json_path = output_dir / "audit_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        text_path = output_dir / "audit_summary.txt"
        self._generate_text_summary(report_data, text_path)
        
        html_path = output_dir / "audit_report.html"
        self._generate_html_report(report_data, html_path)
        
        logger.info(f"Report generated: {output_dir}")
        
        return {
            'json_report': str(json_path),
            'text_summary': str(text_path),
            'html_report': str(html_path)
        }
    
    def _generate_recommendations(
        self,
        quality_checks: Dict,
        anomaly_results: Dict,
        scores: Dict
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        missing_check = quality_checks.get('missing_values_check', {})
        if missing_check.get('severity') in ['high', 'medium']:
            recommendations.append(
                "Apply imputation strategies for missing values (KNN, median, or mode imputation)"
            )
        
        dup_check = quality_checks.get('duplicates_check', {})
        if dup_check.get('severity') in ['high', 'medium']:
            recommendations.append(
                "Remove duplicate records before model training to avoid data leakage"
            )
        
        balance_check = quality_checks.get('class_balance_check', {})
        if balance_check.get('severity') in ['high', 'medium']:
            imbalance_ratio = balance_check.get('imbalance_ratio', 0)
            if imbalance_ratio < 10:
                recommendations.extend([
                    f"Mild class imbalance ({imbalance_ratio:.1f}:1) is common in real-world datasets",
                    "  Optional: Apply SMOTE, class weights, or stratified sampling if needed"
                ])
            else:
                recommendations.append(
                    f"Address significant class imbalance ({imbalance_ratio:.1f}:1 ratio) using "
                    f"advanced techniques (SMOTE, ensemble methods, or cost-sensitive learning)"
                )
        
        if 'combined' in anomaly_results:
            anomaly_pct = anomaly_results['combined']['anomaly_percentage']
            n_anomalies = anomaly_results['combined']['n_anomalies']
            
            if anomaly_pct > 15:
                recommendations.extend([
                    f"{n_anomalies} statistical outliers detected ({anomaly_pct:.1f}%) by ML models",
                    "  These are model-flagged anomalies, not confirmed errors",
                    "  Recommended: Manual review to distinguish legitimate edge cases from data errors",
                    "  Consider: Domain expert validation for high-impact decisions"
                ])
            elif anomaly_pct > 5:
                recommendations.extend([
                    f"{n_anomalies} statistical outliers detected ({anomaly_pct:.1f}%)--typical for real-world data",
                    "  Most are likely legitimate extreme values or rare combinations",
                    "  Optional: Review top outliers for domain-specific relevance"
                ])
            else:
                recommendations.append(
                    f"Low outlier rate ({anomaly_pct:.1f}%)--within expected statistical variation"
                )
        
        overall_score = scores['overall_score']
        
        if overall_score >= 90:
            recommendations.append(
                "Dataset quality is excellent--proceed with model training confidently"
            )
        elif overall_score >= 80:
            recommendations.append(
                "Dataset quality is very good--suitable for immediate use with minor preprocessing"
            )
        elif overall_score >= 70:
            recommendations.append(
                "Dataset quality is good--address minor issues for optimal model performance"
            )
        elif overall_score >= 60:
            recommendations.append(
                "Dataset requires targeted cleaning based on findings above before training"
            )
        else:
            recommendations.append(
                "Major quality issues detected--comprehensive data cleaning required"
            )
        
        if not recommendations:
            recommendations.append(
                "Dataset quality is acceptable--ready for modeling with standard preprocessing"
            )
        
        return recommendations
    
    def _generate_text_summary(self, report_data: Dict, output_path: str):
        """Generate text summary report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("DATASET QUALITY AUDIT REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            metadata = report_data['metadata']
            f.write(f"Dataset: {metadata['file_name']}\n")
            f.write(f"Type: {metadata['dataset_type']}\n")
            f.write(f"Samples: {metadata['n_samples']}\n")
            f.write(f"Features: {metadata.get('n_features', 'N/A')}\n")
            f.write(f"Timestamp: {report_data['timestamp']}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("QUALITY SCORES\n")
            f.write("-" * 70 + "\n")
            scores = report_data['scores']
            f.write(f"Overall Score: {scores['overall_score']}/100 ({scores['rating']})\n\n")
            
            f.write("Component Scores (Weighted):\n")
            weighted_scores = {
                comp: round(score * scores['weights'].get(comp, 0.25), 2)
                for comp, score in scores['component_scores'].items()
            }
            
            for component, weighted_score in weighted_scores.items():
                percentage = scores['component_scores'][component]
                f.write(f"  - {component.capitalize()}: {weighted_score}/25 ({percentage:.0f}%)\n")
            
            f.write("\nScore Calculation:\n")
            calc_parts = " + ".join([f"{v:.2f}" for v in weighted_scores.values()])
            total = sum(weighted_scores.values())
            f.write(f"  {calc_parts} = {total:.2f}/100\n")
            
            f.write(f"\nRecommendation: {scores['recommendation']}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("DETECTED ISSUES\n")
            f.write("-" * 70 + "\n")
            
            quality_checks = report_data['quality_checks']
            
            missing_check = quality_checks.get('missing_values_check', {})
            f.write(
                f"Missing Values: {missing_check.get('missing_percentage', 0):.2f}% "
                f"[{missing_check.get('status', 'unknown').upper()}]\n"
            )
            
            dup_check = quality_checks.get('duplicates_check', {})
            f.write(
                f"Duplicates: {dup_check.get('duplicate_percentage', 0):.2f}% "
                f"[{dup_check.get('status', 'unknown').upper()}]\n"
            )
            
            if 'combined' in report_data['anomaly_results']:
                anomaly_res = report_data['anomaly_results']['combined']
                f.write(
                    f"Statistical Outliers: {anomaly_res['n_anomalies']} "
                    f"({anomaly_res['anomaly_percentage']:.2f}%)\n"
                )
                f.write(
                    "  Note: ML-based detection; not necessarily data errors\n"
                )
            
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 70 + "\n")
            for rec in report_data['recommendations']:
                f.write(f"{rec}\n")
            
            f.write("\n" + "=" * 70 + "\n")
    
    def _generate_html_report(self, report_data: Dict, output_path: str):
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Dataset Quality Audit Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 40px;
                    background: #f5f5f5;
                }
                .container {
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    max-width: 1000px;
                    margin: 0 auto;
                }
                h1 {
                    color: #333;
                    border-bottom: 3px solid #4CAF50;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #555;
                    margin-top: 30px;
                    border-bottom: 2px solid #ddd;
                    padding-bottom: 8px;
                }
                .score {
                    font-size: 48px;
                    font-weight: bold;
                    color: {{ score_color }};
                    margin: 20px 0;
                }
                .rating {
                    font-size: 24px;
                    color: #666;
                    margin-bottom: 10px;
                }
                .section {
                    margin: 20px 0;
                    padding: 20px;
                    background: #f9f9f9;
                    border-radius: 5px;
                    border-left: 4px solid #4CAF50;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background: #4CAF50;
                    color: white;
                    font-weight: 600;
                }
                .recommendation {
                    background: #e3f2fd;
                    padding: 15px;
                    margin: 8px 0;
                    border-left: 4px solid #2196F3;
                    border-radius: 4px;
                }
                .note {
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 12px;
                    margin: 10px 0;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Dataset Quality Audit Report</h1>
                
                <div class="section">
                    <h2>Dataset Information</h2>
                    <p><strong>Name:</strong> {{ metadata.file_name }}</p>
                    <p><strong>Type:</strong> {{ metadata.dataset_type }}</p>
                    <p><strong>Samples:</strong> {{ metadata.n_samples }}</p>
                    <p><strong>Features:</strong> {{ metadata.get('n_features', 'N/A') }}</p>
                    <p><strong>Audit Date:</strong> {{ timestamp }}</p>
                </div>
                
                <div class="section">
                    <h2>Quality Score</h2>
                    <div class="score">{{ scores.overall_score }}/100</div>
                    <div class="rating">Rating: {{ scores.rating }}</div>
                    <p><strong>Recommendation:</strong> {{ scores.recommendation }}</p>
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    {% for rec in recommendations %}
                    <div class="recommendation">{{ rec }}</div>
                    {% endfor %}
                </div>
            </div>
        </body>
        </html>
        """
        
        score = report_data['scores']['overall_score']
        if score >= 90:
            score_color = '#4CAF50'
        elif score >= 75:
            score_color = '#8BC34A'
        elif score >= 60:
            score_color = '#FFC107'
        else:
            score_color = '#F44336'
        
        template = Template(html_template)
        html_content = template.render(
            metadata=report_data['metadata'],
            scores=report_data['scores'],
            recommendations=report_data['recommendations'],
            timestamp=report_data['timestamp'],
            score_color=score_color
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
