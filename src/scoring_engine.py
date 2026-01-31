from typing import Dict
from utils.logger import default_logger as logger


class ScoringEngine:
    """Computes comprehensive dataset health score"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.scoring_config = self.config.get('scoring', {})
        
        self.weights = {
            'completeness': self.scoring_config.get('completeness_weight', 0.25),
            'consistency': self.scoring_config.get('consistency_weight', 0.25),
            'balance': self.scoring_config.get('balance_weight', 0.25),
            'integrity': self.scoring_config.get('integrity_weight', 0.25)
        }
        
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def calculate_score(self, profile: Dict, quality_checks: Dict, anomaly_results: Dict) -> Dict:
        """Calculate overall dataset health score"""
        logger.info("Calculating dataset health score...")
        
        scores = {
            'completeness': self._score_completeness(profile, quality_checks),
            'consistency': self._score_consistency(profile, quality_checks),
            'balance': self._score_balance(quality_checks),
            'integrity': self._score_integrity(anomaly_results, quality_checks)
        }
        
        overall_score = sum(
            scores[k] * self.weights[k] for k in scores
        )
        
        rating, recommendation = self._get_rating_and_recommendation(
            overall_score, scores, quality_checks, anomaly_results
        )
        
        result = {
            'overall_score': round(overall_score, 2),
            'component_scores': {k: round(v, 2) for k, v in scores.items()},
            'weights': {k: round(v, 2) for k, v in self.weights.items()},
            'rating': rating,
            'recommendation': recommendation,
            'details': self._get_score_details(scores, quality_checks, anomaly_results)
        }
        
        logger.info(f"Component scores: {result['component_scores']}")
        logger.info(f"Weights: {result['weights']}")
        logger.info(f"Calculation: {self._show_calculation(scores)}")
        logger.info(f"Dataset Health Score: {result['overall_score']}/100 ({rating})")
        
        return result
    
    def _show_calculation(self, scores: Dict) -> str:
        """Show score calculation"""
        calc = " + ".join(
            f"({scores[k]:.1f} × {self.weights[k]:.2f})"
            for k in scores
        )
        total = sum(scores[k] * self.weights[k] for k in scores)
        return f"{calc} = {total:.2f}"
    
    def _score_completeness(self, profile: Dict, quality_checks: Dict) -> float:
        """Score data completeness"""
        missing_pct = quality_checks.get('missing_values_check', {}).get('missing_percentage', 0)
        
        if missing_pct < 1:
            score = 100
        elif missing_pct < 5:
            score = 90
        elif missing_pct < 10:
            score = 75
        elif missing_pct < 20:
            score = 60
        else:
            score = max(0, 60 - (missing_pct - 20) * 2)
        
        return float(score)
    
    def _score_consistency(self, profile: Dict, quality_checks: Dict) -> float:
        """Score data consistency"""
        dup_pct = quality_checks.get('duplicates_check', {}).get('duplicate_percentage', 0)
        
        if dup_pct == 0:
            score = 100
        elif dup_pct < 1:
            score = 95
        elif dup_pct < 5:
            score = 85
        elif dup_pct < 10:
            score = 70
        else:
            score = max(0, 70 - (dup_pct - 10) * 3.5)
        
        return float(score)
    
    def _score_balance(self, quality_checks: Dict) -> float:
        """Score class balance"""
        balance_check = quality_checks.get('class_balance_check', {})
        status = balance_check.get('status', 'not_applicable')
        
        if status == 'not_applicable':
            return 100.0
        
        ratio = balance_check.get('imbalance_ratio', 1.0)
        
        if ratio < 2:
            score = 100
        elif ratio < 5:
            score = 90
        elif ratio < 10:
            score = 75
        elif ratio < 20:
            score = 60
        else:
            score = max(50, 60 - (ratio - 20) * 0.5)
        
        return float(score)
    
    def _score_integrity(self, anomaly_results: Dict, quality_checks: Dict) -> float:
        """Score data integrity"""
        if not anomaly_results or 'combined' not in anomaly_results:
            outlier_pct = quality_checks.get('outliers_check', {}).get('outlier_percentage', 0)
            
            if outlier_pct < 5:
                return 100.0
            elif outlier_pct < 10:
                return 90.0
            elif outlier_pct < 15:
                return 75.0
            else:
                return max(50, 75 - (outlier_pct - 15))
        
        anomaly_pct = anomaly_results.get('combined', {}).get('anomaly_percentage', 0)
        
        if anomaly_pct < 5:
            score = 100
        elif anomaly_pct < 10:
            score = 90
        elif anomaly_pct < 15:
            score = 75
        elif anomaly_pct < 25:
            score = 60
        else:
            score = max(40, 60 - (anomaly_pct - 25) * 1.5)
        
        return float(score)
    
    def _get_rating_and_recommendation(
        self,
        overall_score: float,
        scores: Dict,
        quality_checks: Dict,
        anomaly_results: Dict
    ) -> tuple:
        """Determine rating and recommendation"""
        if overall_score >= 90:
            rating = 'Excellent'
        elif overall_score >= 80:
            rating = 'Very Good'
        elif overall_score >= 70:
            rating = 'Good'
        elif overall_score >= 60:
            rating = 'Acceptable'
        elif overall_score >= 50:
            rating = 'Fair'
        else:
            rating = 'Poor'
        
        recommendation = self._generate_smart_recommendation(
            overall_score, scores, quality_checks, anomaly_results
        )
        
        return rating, recommendation
    
    def _generate_smart_recommendation(
        self,
        overall_score: float,
        scores: Dict,
        quality_checks: Dict,
        anomaly_results: Dict
    ) -> str:
        """Generate recommendation"""
        if overall_score >= 90:
            return "Dataset is production-ready with excellent quality."
        elif overall_score >= 80:
            return "Dataset quality is very good and suitable for immediate use."
        elif overall_score >= 70:
            weakest = min(scores.items(), key=lambda x: x[1])[0]
            
            if weakest == 'balance':
                ratio = quality_checks.get('class_balance_check', {}).get('imbalance_ratio', 1)
                return f"Class imbalance detected ({ratio:.1f}:1). Apply class weights or resampling."
            elif weakest == 'completeness':
                pct = quality_checks.get('missing_values_check', {}).get('missing_percentage', 0)
                return f"{pct:.1f}% missing values detected. Apply imputation."
            elif weakest == 'consistency':
                pct = quality_checks.get('duplicates_check', {}).get('duplicate_percentage', 0)
                return f"{pct:.1f}% duplicates detected. Deduplication recommended."
            else:
                return "Review detected anomalies to separate valid outliers from errors."
        
        elif overall_score >= 60:
            return "Dataset usable but requires cleaning before training."
        elif overall_score >= 50:
            return "Dataset has multiple quality issues requiring attention."
        else:
            return "Major quality issues detected. Not suitable for immediate training."
    
    def _get_score_details(
        self,
        scores: Dict,
        quality_checks: Dict,
        anomaly_results: Dict
    ) -> Dict:
        """Detailed score breakdown"""
        return {
            'completeness': {
                'score': round(scores['completeness'], 2),
                'missing_pct': quality_checks.get('missing_values_check', {}).get('missing_percentage', 0),
                'interpretation': self._interpret_completeness(scores['completeness'])
            },
            'consistency': {
                'score': round(scores['consistency'], 2),
                'duplicate_pct': quality_checks.get('duplicates_check', {}).get('duplicate_percentage', 0),
                'interpretation': self._interpret_consistency(scores['consistency'])
            },
            'balance': {
                'score': round(scores['balance'], 2),
                'imbalance_ratio': quality_checks.get('class_balance_check', {}).get('imbalance_ratio'),
                'interpretation': self._interpret_balance(scores['balance'])
            },
            'integrity': {
                'score': round(scores['integrity'], 2),
                'anomaly_pct': anomaly_results.get('combined', {}).get('anomaly_percentage', 0),
                'interpretation': self._interpret_integrity(scores['integrity'])
            }
        }
    
    def _interpret_completeness(self, score: float) -> str:
        if score >= 90:
            return "Excellent data coverage"
        elif score >= 75:
            return "Low missing values"
        elif score >= 60:
            return "Moderate missing values"
        else:
            return "High missing values"
    
    def _interpret_consistency(self, score: float) -> str:
        if score >= 95:
            return "No duplicates"
        elif score >= 85:
            return "Minor duplicates"
        elif score >= 70:
            return "Deduplication recommended"
        else:
            return "Significant duplicates"
    
    def _interpret_balance(self, score: float) -> str:
        if score >= 95:
            return "Well-balanced classes"
        elif score >= 75:
            return "Mild imbalance"
        elif score >= 60:
            return "Moderate imbalance"
        else:
            return "Severe imbalance"
    
    def _interpret_integrity(self, score: float) -> str:
        if score >= 90:
            return "Minimal anomalies"
        elif score >= 75:
            return "Some anomalies"
        elif score >= 60:
            return "Notable anomalies"
        else:
            return "High anomaly rate"
