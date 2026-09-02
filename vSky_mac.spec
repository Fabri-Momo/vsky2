# -*- mode: python ; coding: utf-8 -*-
# vSky PyInstaller spec file for macOS (.app bundle)
# Usage (from macOS):
#     conda activate vsky
#     python -m PyInstaller --noconfirm --clean vSky_mac.spec
#
# Notes:
# - CUDA/CuPy is not available on macOS.
# - GPU support on Apple Silicon uses PyTorch MPS (Metal Performance Shaders).
# - GDAL and PROJ data directories are expected from a conda-forge env.
# - The generated output is dist/vSky/vSky.app

import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

block_cipher = None
project_dir = SPECPATH
env_path = os.environ.get('CONDA_PREFIX', os.path.dirname(os.path.dirname(sys.executable)))

# GDAL/PROJ data directories (conda-forge layout on macOS: $CONDA_PREFIX/share)
gdal_data = os.path.join(env_path, 'share', 'gdal')
proj_data = os.path.join(env_path, 'share', 'proj')

datas = []
if os.path.isdir(gdal_data):
    datas.append((gdal_data, 'gdal_data'))
if os.path.isdir(proj_data):
    datas.append((proj_data, 'proj_data'))

# Resources needed by the application
datas.append((os.path.join(project_dir, 'resources'), 'resources'))
datas.append((os.path.join(project_dir, 'qrc_resources.py'), '.'))

# osgeo may require additional data files / dylibs
datas += collect_data_files('osgeo', include_py_files=False)
binaries = collect_dynamic_libs('osgeo')

# PyTorch (used for Apple Silicon MPS GPU support)
try:
    import torch
    datas += collect_data_files('torch', include_py_files=False)
    binaries += collect_dynamic_libs('torch')
    hiddenimports_torch = [
        'torch', 'torch._C', 'torch._C._nn', 'torch._C._autograd',
        'torch.backends', 'torch.backends.mps', 'torch.mps',
    ]
    print(f"PyTorch {torch.__version__} found, including MPS support")
except ImportError:
    hiddenimports_torch = []
    print("PyTorch not found, building without MPS GPU support")

hiddenimports = [
    'osgeo', 'osgeo.gdal', 'osgeo.osr', 'osgeo.ogr',
    'osgeo._gdal', 'osgeo._osr', 'osgeo._ogr',
    'osgeo._gdalconst', 'osgeo._gdal_array',
    'numpy', 'scipy', 'scipy.signal',
    'PIL', 'PIL.Image',
    'qrc_resources',
] + hiddenimports_torch

a = Analysis(
    [os.path.join(project_dir, 'vSky2.py')],
    pathex=[project_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(project_dir, '_runtime_hook.py')],
    excludes=['tkinter', 'cupy', 'cupy_backends', 'nvidia'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# macOS icon (convert icon_vSky.ico to icon_vSky.icns first)
icon_path = os.path.join(project_dir, 'vSky2.icns')
if not os.path.isfile(icon_path):
    icon_path = None
    print("WARNING: vSky2.icns not found. Create it from the vSky2 PNGs.")

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='vSky',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,      # required for macOS .app argument handling
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='vSky',
)

app = BUNDLE(
    coll,
    name='vSky.app',
    icon=icon_path,
    bundle_identifier='fr.u-bourgogne.vsky',
)
