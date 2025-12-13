"""
signals.py
Validation utilities for m02_extract_signals output.

Provides functions to validate that signal extraction worked correctly:
- Check signal counts
- Verify gap detection
- Compare v0 vs variations
- Validate Grad-CAM files
"""

import json
import sqlite3
from pathlib import Path


def validate_signals(db_path: Path, gradcam_dir: Path) -> dict:
    """
    Run all validation checks on extracted signals.

    Returns dict with validation results and pass/fail status.
    """
    results = {
        "checks": {},
        "passed": True,
        "summary": ""
    }

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # =========================================================================
    # Check 1: Signal count
    # =========================================================================
    cursor.execute("SELECT COUNT(*) FROM signals")
    signal_count = cursor.fetchone()[0]
    results["checks"]["signal_count"] = {
        "value": signal_count,
        "expected": "80 (or matching image count)",
        "pass": signal_count > 0
    }

    # =========================================================================
    # Check 2: Gap detection worked (variations should have gaps)
    # =========================================================================
    cursor.execute("""
        SELECT image_id, num_gaps, gaps_json
        FROM signals
        WHERE num_gaps > 0
        LIMIT 5
    """)
    gaps_found = cursor.fetchall()
    results["checks"]["gaps_detected"] = {
        "count": len(gaps_found),
        "samples": [
            {"image_id": row[0], "num_gaps": row[1], "gaps": json.loads(row[2]) if row[2] else []}
            for row in gaps_found
        ],
        "pass": len(gaps_found) > 0
    }

    # =========================================================================
    # Check 3: v0_original has fewer gaps than variations
    # =========================================================================
    cursor.execute("""
        SELECT
            CASE WHEN image_id LIKE '%v0%' THEN 'v0_original' ELSE 'variation' END as type,
            COUNT(*) as count,
            AVG(num_gaps) as avg_gaps,
            AVG(avg_object_score) as avg_score
        FROM signals
        GROUP BY type
    """)
    type_comparison = cursor.fetchall()
    type_stats = {}
    for row in type_comparison:
        type_stats[row[0]] = {
            "count": row[1],
            "avg_gaps": round(row[2], 3) if row[2] else 0,
            "avg_score": round(row[3], 4) if row[3] else 0
        }

    v0_gaps = type_stats.get("v0_original", {}).get("avg_gaps", 0)
    var_gaps = type_stats.get("variation", {}).get("avg_gaps", 0)

    results["checks"]["v0_vs_variations"] = {
        "v0_original": type_stats.get("v0_original", {}),
        "variation": type_stats.get("variation", {}),
        "pass": v0_gaps <= var_gaps  # v0 should have same or fewer gaps
    }

    # =========================================================================
    # Check 4: Specific gap types (color mismatch)
    # =========================================================================
    cursor.execute("""
        SELECT image_id, gaps_json
        FROM signals
        WHERE gaps_json LIKE '%wrong_color%'
        LIMIT 5
    """)
    color_gaps = cursor.fetchall()
    results["checks"]["wrong_color_gaps"] = {
        "count": len(color_gaps),
        "samples": [
            {"image_id": row[0], "gaps": json.loads(row[1]) if row[1] else []}
            for row in color_gaps
        ],
        "pass": True  # Informational, not a strict requirement
    }

    # Check for missing object gaps
    cursor.execute("""
        SELECT image_id, gaps_json
        FROM signals
        WHERE gaps_json LIKE '%missing%'
        LIMIT 5
    """)
    missing_gaps = cursor.fetchall()
    results["checks"]["missing_object_gaps"] = {
        "count": len(missing_gaps),
        "samples": [
            {"image_id": row[0], "gaps": json.loads(row[1]) if row[1] else []}
            for row in missing_gaps
        ],
        "pass": True  # Informational
    }

    # =========================================================================
    # Check 5: Grad-CAM files exist
    # =========================================================================
    gradcam_files = list(gradcam_dir.glob("*.npy")) if gradcam_dir.exists() else []
    results["checks"]["gradcam_files"] = {
        "count": len(gradcam_files),
        "expected": signal_count,
        "pass": len(gradcam_files) >= signal_count * 0.9  # Allow 10% tolerance
    }

    # =========================================================================
    # Check 6: Objects detected per image
    # =========================================================================
    cursor.execute("SELECT AVG(num_objects) FROM signals")
    avg_objects = cursor.fetchone()[0]
    results["checks"]["avg_objects_per_image"] = {
        "value": round(avg_objects, 2) if avg_objects else 0,
        "expected": "> 2",
        "pass": (avg_objects or 0) > 2
    }

    conn.close()

    # =========================================================================
    # Overall pass/fail
    # =========================================================================
    critical_checks = ["signal_count", "gaps_detected", "gradcam_files"]
    results["passed"] = all(
        results["checks"][c]["pass"] for c in critical_checks
    )

    return results


def print_validation_report(results: dict) -> None:
    """Print formatted validation report."""
    print("\n" + "=" * 70)
    print("SIGNAL EXTRACTION VALIDATION REPORT")
    print("=" * 70)

    checks = results["checks"]

    # 1. Signal count
    c = checks["signal_count"]
    status = "PASS" if c["pass"] else "FAIL"
    print(f"\n[{status}] Signal Count: {c['value']} (expected: {c['expected']})")

    # 2. Gaps detected
    c = checks["gaps_detected"]
    status = "PASS" if c["pass"] else "FAIL"
    print(f"\n[{status}] Gaps Detected: {c['count']} images with gaps")
    if c["samples"]:
        print("  Sample gaps found:")
        for s in c["samples"][:3]:
            gaps_summary = ", ".join([
                f"{g['object_noun']}:{g['issue']}"
                for g in s["gaps"][:2]
            ]) if s["gaps"] else "none"
            print(f"    {s['image_id']}: {s['num_gaps']} gaps - {gaps_summary}")

    # 3. v0 vs variations
    c = checks["v0_vs_variations"]
    status = "PASS" if c["pass"] else "WARN"
    print(f"\n[{status}] v0 vs Variations Comparison:")
    if c.get("v0_original"):
        v0 = c["v0_original"]
        print(f"  v0_original: {v0['count']} images, avg_gaps={v0['avg_gaps']}, avg_score={v0['avg_score']}")
    if c.get("variation"):
        var = c["variation"]
        print(f"  variations:  {var['count']} images, avg_gaps={var['avg_gaps']}, avg_score={var['avg_score']}")

    # 4. Wrong color gaps
    c = checks["wrong_color_gaps"]
    print(f"\n[INFO] Wrong Color Gaps: {c['count']} images")
    if c["samples"]:
        for s in c["samples"][:2]:
            for g in s["gaps"]:
                if g.get("issue") == "wrong_color":
                    print(f"  {s['image_id']}: expected={g.get('expected')}, found={g.get('found')}")

    # 5. Missing object gaps
    c = checks["missing_object_gaps"]
    print(f"\n[INFO] Missing Object Gaps: {c['count']} images")
    if c["samples"]:
        for s in c["samples"][:2]:
            for g in s["gaps"]:
                if g.get("issue") == "missing":
                    print(f"  {s['image_id']}: missing {g.get('object_noun')}")

    # 6. Grad-CAM files
    c = checks["gradcam_files"]
    status = "PASS" if c["pass"] else "FAIL"
    print(f"\n[{status}] Grad-CAM Files: {c['count']} / {c['expected']} expected")

    # 7. Average objects
    c = checks["avg_objects_per_image"]
    status = "PASS" if c["pass"] else "WARN"
    print(f"\n[{status}] Avg Objects Per Image: {c['value']} (expected: {c['expected']})")

    # Overall
    print("\n" + "=" * 70)
    overall = "PASSED" if results["passed"] else "FAILED"
    print(f"OVERALL: {overall}")
    print("=" * 70 + "\n")


def run_validation(db_path: Path, gradcam_dir: Path) -> bool:
    """
    Run validation and print report.

    Returns True if all critical checks passed.
    """
    results = validate_signals(db_path, gradcam_dir)
    print_validation_report(results)
    return results["passed"]


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "outputs" / "centralized.db"
    gradcam_dir = project_root / "outputs" / "m02_gradcam"

    if not db_path.exists():
        print(f"Database not found: {db_path}")
    else:
        run_validation(db_path, gradcam_dir)
