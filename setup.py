"""
setup.py - Build vSky as a standalone executable using PyInstaller.

Usage:
    conda activate vsky
    python setup.py

Uses the vSky_build.spec file which handles:
- GDAL/PROJ data files and DLL resolution via pathex
- CuPy/CUDA GPU support with collect_submodules
- NVIDIA CUDA NVRTC DLLs
- Filtering of conflicting DLLs (CRT/VC runtime, old CUDA 11)
- Runtime hook for GDAL_DATA and PROJ_LIB environment variables
"""

import os
import sys
import subprocess


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Choose the right PyInstaller spec for the platform
    if sys.platform == 'win32':
        spec_file = os.path.join(project_dir, 'vSky_build.spec')
    elif sys.platform == 'darwin':
        spec_file = os.path.join(project_dir, 'vSky_mac.spec')
    elif sys.platform.startswith('linux'):
        spec_file = os.path.join(project_dir, 'vSky_linux.spec')
    else:
        print(f"ERROR: Unsupported platform for bundled build: {sys.platform}")
        print("You can still run vSky2.py from source.")
        sys.exit(1)

    if not os.path.isfile(spec_file):
        print(f"ERROR: Spec file not found: {spec_file}")
        sys.exit(1)

    # Check PyInstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])

    print(f"Project directory: {project_dir}")
    print(f"Spec file: {spec_file}")
    print(f"Conda env: {os.environ.get('CONDA_PREFIX', 'N/A')}")

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--noconfirm',
        '--clean',
        spec_file,
    ]

    print("\n" + "=" * 60)
    print("Running PyInstaller with vSky_build.spec")
    print("=" * 60 + "\n")

    result = subprocess.run(cmd, cwd=project_dir)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("BUILD SUCCESSFUL!")
        if sys.platform == 'win32':
            print("Executable: dist/vSky2/vSky2.exe")
        elif sys.platform == 'darwin':
            print("Application bundle: dist/vSky2.app")
        elif sys.platform.startswith('linux'):
            print("Executable: dist/vSky2/vSky2")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("BUILD FAILED!")
        print(f"Exit code: {result.returncode}")
        print("=" * 60)
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()
