#!/usr/bin/env python3
"""
Generate Python stub files for the libmatmul C++ extension
"""
import os
import sys
import shutil
import subprocess
import re
from pathlib import Path


def process_stub_file(stub_file: Path) -> None:
    """
    Process the generated stub file, replacing numpy.ndarray[numpy.int32] with NDArray[np.int32]
    """
    with open(stub_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update import statements
    # Remove original numpy import
    content = re.sub(r'^import numpy\s*$', '', content, flags=re.MULTILINE)

    # Add new imports after 'from __future__ import annotations'
    import_pattern = r'(from __future__ import annotations\n)'
    new_imports = r'\1from numpy.typing import NDArray\nimport numpy as np\n'
    content = re.sub(import_pattern, new_imports, content)

    # Replace all numpy.ndarray[numpy.int32] with NDArray[np.int32]
    content = re.sub(
        r'numpy\.ndarray\[numpy\.int32\]', 'NDArray[np.int32]', content)

    # Write back to file
    with open(stub_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Processed stub file: {stub_file}")


def main():
    # Project root directory
    project_root = Path(__file__).parent.parent
    benchmark_dir = project_root / "benchmark"

    # Check if libmatmul.so exists
    libmatmul_so = benchmark_dir / "libmatmul.so"
    if not libmatmul_so.exists():
        print(f"Error: {libmatmul_so} does not exist")
        print("Please build the project first: xmake build")
        sys.exit(1)

    # Change to benchmark directory
    os.chdir(benchmark_dir)

    # Remove old stubs directory
    stubs_dir = benchmark_dir / "stubs"
    if stubs_dir.exists():
        shutil.rmtree(stubs_dir)
        print("✓ Removed old stubs directory")

    # Run pybind11-stubgen
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(benchmark_dir)

        result = subprocess.run([
            sys.executable, "-m", "pybind11_stubgen",
            "--output-dir", "stubs",
            "libmatmul"
        ], check=True, capture_output=True, text=True, env=env)
        print("✓ pybind11-stubgen completed successfully")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: pybind11-stubgen failed")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)

    # Process the generated stub file
    stub_src = benchmark_dir / "stubs" / "libmatmul.pyi"
    if stub_src.exists():
        process_stub_file(stub_src)
    else:
        print(f"Error: Generated stub file does not exist: {stub_src}")
        sys.exit(1)

    # Copy processed stub file to project root
    stub_dst = project_root / "libmatmul.pyi"
    shutil.copy2(stub_src, stub_dst)
    print(f"✓ Copied stub file to {stub_dst}")

    print("✓ Stub file generation and processing completed!")


if __name__ == "__main__":
    main()
