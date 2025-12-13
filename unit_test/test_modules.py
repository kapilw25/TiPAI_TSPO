"""
test_modules.py
Validation tests for all POC modules.
Tests: py_compile, AST syntax, imports, function signatures
"""

import ast
import sys
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
UTILS_DIR = SRC_DIR / "utils"

MODULES = [
    "m01_generate_images",
    "m02_clip_scoring",
    "m03_create_heatmaps",
    "m04_create_pairs",
    "m05_demo_app",
]

UTILS_MODULES = [
    "hf_utils",
]


def test_py_compile(module_name: str) -> tuple:
    """Test if module compiles without syntax errors."""
    import py_compile
    module_path = SRC_DIR / f"{module_name}.py"

    try:
        py_compile.compile(str(module_path), doraise=True)
        return True, "OK"
    except py_compile.PyCompileError as e:
        return False, str(e)


def test_ast_parse(module_name: str) -> tuple:
    """Test if module can be parsed as valid AST."""
    module_path = SRC_DIR / f"{module_name}.py"

    try:
        with open(module_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def test_imports(module_name: str) -> tuple:
    """Test if all imports are valid (without executing)."""
    module_path = SRC_DIR / f"{module_name}.py"

    try:
        with open(module_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])

        # Check standard library and known packages
        stdlib = {'json', 'sqlite3', 'pathlib', 'os', 'sys', 'argparse', 'typing', 'datetime'}
        known_packages = {'torch', 'PIL', 'tqdm', 'diffusers', 'open_clip', 'numpy',
                         'matplotlib', 'seaborn', 'scipy', 'gradio', 'cv2', 'transformers',
                         'accelerate', 'torchvision', 'datasets', 'huggingface_hub'}

        missing = []
        for imp in set(imports):
            if imp not in stdlib and imp not in known_packages:
                # Check if it's a local module or utils package
                if imp == "utils":
                    continue  # utils is our local package
                if not (SRC_DIR / f"{imp}.py").exists():
                    missing.append(imp)

        if missing:
            return False, f"Unknown imports: {missing}"
        return True, f"OK ({len(set(imports))} imports)"
    except Exception as e:
        return False, str(e)


def test_functions(module_name: str) -> tuple:
    """Check that expected functions exist."""
    module_path = SRC_DIR / f"{module_name}.py"

    expected_functions = {
        "m01_generate_images": ["load_prompts", "load_pipeline", "generate_image", "generate_all_images"],
        "m02_clip_scoring": ["load_clip_model", "compute_global_score", "compute_patch_scores", "score_all_images"],
        "m03_create_heatmaps": ["create_heatmap_overlay", "generate_all_heatmaps"],
        "m04_create_pairs": ["create_comparison_image", "create_all_pairs"],
        "m05_demo_app": ["browse_pairs", "create_heatmap", "main"],
    }

    try:
        with open(module_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)
        defined_functions = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_functions.add(node.name)

        expected = set(expected_functions.get(module_name, []))
        missing = expected - defined_functions

        if missing:
            return False, f"Missing functions: {missing}"
        return True, f"OK ({len(defined_functions)} functions)"
    except Exception as e:
        return False, str(e)


def test_config_class(module_name: str) -> tuple:
    """Check that Config class exists with required attributes."""
    module_path = SRC_DIR / f"{module_name}.py"

    try:
        with open(module_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)
        has_config = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Config":
                has_config = True
                break

        if not has_config:
            return False, "Config class not found"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_no_hardcoded_paths(module_name: str) -> tuple:
    """Check for hardcoded absolute paths (bad practice)."""
    module_path = SRC_DIR / f"{module_name}.py"

    try:
        with open(module_path, 'r') as f:
            source = f.read()

        # Look for hardcoded paths like /Users/, /home/, C:\
        bad_patterns = ['/Users/', '/home/', 'C:\\', 'D:\\']
        found = []

        for i, line in enumerate(source.split('\n'), 1):
            for pattern in bad_patterns:
                if pattern in line and not line.strip().startswith('#'):
                    found.append(f"Line {i}: {pattern}")

        if found:
            return False, f"Hardcoded paths: {found[:3]}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_py_compile_utils(module_name: str) -> tuple:
    """Test if utils module compiles without syntax errors."""
    import py_compile
    module_path = UTILS_DIR / f"{module_name}.py"

    try:
        py_compile.compile(str(module_path), doraise=True)
        return True, "OK"
    except py_compile.PyCompileError as e:
        return False, str(e)


def test_ast_parse_utils(module_name: str) -> tuple:
    """Test if utils module can be parsed as valid AST."""
    module_path = UTILS_DIR / f"{module_name}.py"

    try:
        with open(module_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def run_all_tests():
    """Run all tests on all modules."""
    print("=" * 70)
    print("TiPAI POC Module Validation Tests")
    print("=" * 70)

    all_passed = True
    results = {}

    tests = [
        ("py_compile", test_py_compile),
        ("AST parse", test_ast_parse),
        ("imports", test_imports),
        ("functions", test_functions),
        ("Config class", test_config_class),
        ("no hardcoded paths", test_no_hardcoded_paths),
    ]

    for module in MODULES:
        print(f"\n{module}.py:")
        results[module] = {}

        for test_name, test_func in tests:
            passed, msg = test_func(module)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {test_name}: {msg}")
            results[module][test_name] = (passed, msg)

            if not passed:
                all_passed = False

    # Test utils modules
    print("\n--- Utils Modules ---")
    utils_tests = [
        ("py_compile", test_py_compile_utils),
        ("AST parse", test_ast_parse_utils),
    ]

    for module in UTILS_MODULES:
        print(f"\nutils/{module}.py:")
        results[f"utils/{module}"] = {}

        for test_name, test_func in utils_tests:
            passed, msg = test_func(module)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {test_name}: {msg}")
            results[f"utils/{module}"][test_name] = (passed, msg)

            if not passed:
                all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Fix before deploying to GPU")
    print("=" * 70)

    return all_passed, results


if __name__ == "__main__":
    passed, _ = run_all_tests()
    sys.exit(0 if passed else 1)
