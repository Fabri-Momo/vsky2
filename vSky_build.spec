# -*- mode: python ; coding: utf-8 -*-
# vSky PyInstaller spec file
# Usage: conda activate vsky && pyinstaller vSky_build.spec

import os
import sys
import glob
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None
project_dir = SPECPATH
env_path = os.environ.get('CONDA_PREFIX', os.path.dirname(os.path.dirname(sys.executable)))

# GDAL/PROJ data directories
gdal_data = os.path.join(env_path, 'Library', 'share', 'gdal')
proj_data = os.path.join(env_path, 'Library', 'share', 'proj')

datas = []
if os.path.isdir(gdal_data):
    datas.append((gdal_data, 'gdal_data'))
if os.path.isdir(proj_data):
    datas.append((proj_data, 'proj_data'))

# CuPy needs its C++ include headers (including CCCL/Thrust/CUB/libcudacxx)
# for JIT CUDA kernel compilation at runtime
datas += collect_data_files('cupy')
datas += collect_data_files('taichi', include_py_files=True)

# Include all nvidia CUDA DLLs for CuPy GPU support
binaries = []
try:
    import nvidia
    nv_base = os.path.dirname(nvidia.__path__[0])
    nv_pkg_dir = os.path.join(nv_base, 'nvidia')
    NVIDIA_LIBS = ['cuda_nvrtc', 'cuda_runtime', 'cublas', 'cusolver',
                   'cusparse', 'cufft', 'nvjitlink']
    for lib in NVIDIA_LIBS:
        for subdir in ['bin', 'lib']:
            lib_dir = os.path.join(nv_pkg_dir, lib, subdir)
            if os.path.isdir(lib_dir):
                for dll in glob.glob(os.path.join(lib_dir, '*.dll')):
                    binaries.append((dll, f'nvidia/{lib}/{subdir}'))
    print(f"Found {len(binaries)} NVIDIA CUDA DLLs")
except ImportError:
    print("nvidia packages not found, building without GPU support")

a = Analysis(
    [os.path.join(project_dir, 'vSky2.py')],
    pathex=[
        project_dir,
        os.path.join(env_path, 'Library', 'bin'),  # GDAL/MKL DLLs
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'osgeo', 'osgeo.gdal', 'osgeo.osr', 'osgeo.ogr',
        'osgeo._gdal', 'osgeo._osr', 'osgeo._ogr',
        'osgeo._gdalconst', 'osgeo._gdal_array',
        'numpy', 'scipy', 'scipy.signal',
        'PIL', 'PIL.Image',
        'qrc_resources',
        'nvidia.cuda_nvrtc',
    ] + collect_submodules('cupy') + collect_submodules('cupy_backends') + collect_submodules('taichi'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(project_dir, '_runtime_hook.py')],
    excludes=['tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out DLLs that cause conflicts or are not needed
EXCLUDE_DLL_PREFIXES = (
    # Old CUDA 11 DLLs (conflict with CUDA 12 from CuPy)
    'cublas64_11', 'cublaslt64_11', 'cusolver64_11', 'cufft64_10',
    # Tkinter - excluded
    'tcl', 'tk8',
    # MPI / TBB - not needed
    'msmpi', 'hwloc', 'tbbbind',
    # Windows CRT/VC runtime - use system versions (Win10+), conda versions cause 0xC0000005
    'api-ms-win-',
    'ucrtbase',
    'vcruntime140',
    'msvcp140',
    'concrt140',
    'vcamp140',
    'vccorlib140',
    'vcomp140',
)
a.binaries = [b for b in a.binaries if not any(
    os.path.basename(b[0]).lower().startswith(prefix.lower()) for prefix in EXCLUDE_DLL_PREFIXES
)]
print(f"Binaries after filtering: {len(a.binaries)}")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Icon
icon_path = os.path.join(project_dir, 'vSky2.ico')
if not os.path.isfile(icon_path):
    icon_path = os.path.join(project_dir, 'resources', 'images', 'icon.ico')
if not os.path.isfile(icon_path):
    icon_path = None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='vSky2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # Windowed mode for release
    disable_windowed_traceback=False,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='vSky2',
)
