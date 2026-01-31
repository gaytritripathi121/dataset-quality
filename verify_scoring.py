import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.scoring_engine import ScoringEngine
from utils.helpers import load_config


def test_score_calculation():
    """Test various scoring scenarios"""

    print("=" * 70)
    print("SCORE CALCULATION VERIFICATION TEST")
    print("=" * 70)

    config = load_config("config/config.yaml")
    scorer = ScoringEngine(config)

    print("\n" + "=" * 70)
    print("TEST 1: Perfect Dataset (should be ~95-100)")
    print("=" * 70)

    profile = {}
    quality_checks = {
        "missing_values_check": {
            "status": "excellent",
            "missing_percentage": 0.0,
        },
        "duplicates_check": {
            "status": "excellent",
            "duplicate_percentage": 0.0,
        },
        "class_balance_check": {
            "status": "excellent",
            "imbalance_ratio": 1.2,
        },
    }
    anomaly_results = {
        "combined": {
            "anomaly_percentage": 2.0,
            "n_anomalies": 20,
        }
    }

    result = scorer.calculate_score(profile, quality_checks, anomaly_results)
    print(f"\n✓ Overall Score: {result['overall_score']}/100 ({result['rating']})")
    print(f"✓ Component Scores: {result['component_scores']}")
    print(f"✓ Recommendation: {result['recommendation']}")

    expected = sum(
        result["component_scores"][k] * result["weights"][k]
        for k in result["component_scores"]
    )
    print(
        f"\n✓ Math Check: {expected:.2f} == {result['overall_score']:.2f}? "
        f"{abs(expected - result['overall_score']) < 0.01}"
    )

    print("\n" + "=" * 70)
    print("TEST 2: Your Dataset (Completeness=100%, Consistency=100%, Imbalance=6:1)")
    print("=" * 70)

    quality_checks2 = {
        "missing_values_check": {
            "status": "excellent",
            "missing_percentage": 0.0,
        },
        "duplicates_check": {
            "status": "excellent",
            "duplicate_percentage": 0.0,
        },
        "class_balance_check": {
            "status": "good",
            "imbalance_ratio": 6.0,
        },
    }
    anomaly_results2 = {
        "combined": {
            "anomaly_percentage": 13.5,
            "n_anomalies": 135,
        }
    }

    result2 = scorer.calculate_score(profile, quality_checks2, anomaly_results2)
    print(f"\n✓ Overall Score: {result2['overall_score']}/100 ({result2['rating']})")

    print("✓ Component Scores:")
    for comp, score in result2["component_scores"].items():
        print(f"   - {comp.capitalize()}: {score:.1f}/100")

    print("\n✓ Score Details:")
    for comp, detail in result2["details"].items():
        print(f"   - {comp.capitalize()}: {detail['interpretation']}")

    print(f"\n✓ Recommendation: {result2['recommendation']}")

    expected2 = sum(
        result2["component_scores"][k] * result2["weights"][k]
        for k in result2["component_scores"]
    )
    print(
        f"\n✓ Math Check: {expected2:.2f} == {result2['overall_score']:.2f}? "
        f"{abs(expected2 - result2['overall_score']) < 0.01}"
    )

    print("\n✓ Calculation Breakdown:")
    for comp in result2["component_scores"]:
        score = result2["component_scores"][comp]
        weight = result2["weights"][comp]
        print(f"   {score:.1f} × {weight:.2f} = {score * weight:.2f}")

    print("   " + "-" * 50)
    print(f"   Total: {result2['overall_score']:.2f}/100")

    print("\n" + "=" * 70)
    print("TEST 3: Poor Dataset (should be <50)")
    print("=" * 70)

    quality_checks3 = {
        "missing_values_check": {
            "status": "poor",
            "missing_percentage": 35.0,
        },
        "duplicates_check": {
            "status": "poor",
            "duplicate_percentage": 15.0,
        },
        "class_balance_check": {
            "status": "poor",
            "imbalance_ratio": 50.0,
        },
    }
    anomaly_results3 = {
        "combined": {
            "anomaly_percentage": 40.0,
            "n_anomalies": 400,
        }
    }

    result3 = scorer.calculate_score(profile, quality_checks3, anomaly_results3)
    print(f"\n✓ Overall Score: {result3['overall_score']}/100 ({result3['rating']})")
    print(f"✓ Component Scores: {result3['component_scores']}")
    print(f"✓ Recommendation: {result3['recommendation']}")

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    tests = [
        ("Perfect Dataset", result["overall_score"], 95, 100),
        ("Your Dataset", result2["overall_score"], 70, 85),
        ("Poor Dataset", result3["overall_score"], 0, 50),
    ]

    all_passed = True
    for name, score, min_expected, max_expected in tests:
        passed = min_expected <= score <= max_expected
        print(
            f"{'✓ PASS' if passed else '✗ FAIL'} | "
            f"{name}: {score:.1f} (expected {min_expected}-{max_expected})"
        )
        all_passed &= passed

    print("\n" + "=" * 70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED! Scoring engine is working correctly.")
    else:
        print("✗✗✗ SOME TESTS FAILED! Check the scoring engine.")
    print("=" * 70)


if __name__ == "__main__":
    test_score_calculation()
