import os, sys

# Determine the base path (works for both dev and PyInstaller bundle)
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

if getattr(sys, 'frozen', False):
    # Windows: add base path and bundled nvidia CUDA DLLs for dynamic loading
    if sys.platform == 'win32':
        os.environ['PATH'] = base_path + os.pathsep + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(base_path)

        nv_dir = os.path.join(base_path, 'nvidia')
        if os.path.isdir(nv_dir):
            for lib in os.listdir(nv_dir):
                for subdir in ('bin', 'lib'):
                    dll_dir = os.path.join(nv_dir, lib, subdir)
                    if os.path.isdir(dll_dir):
                        os.environ['PATH'] = dll_dir + os.pathsep + os.environ['PATH']
                        if hasattr(os, 'add_dll_directory'):
                            os.add_dll_directory(dll_dir)

    # Linux: add the bundle directory to the dynamic library search path.
    elif sys.platform.startswith('linux'):
        os.environ['LD_LIBRARY_PATH'] = base_path + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

    # macOS: help the dynamic linker find bundled .dylib libraries.
    # Note: macOS System Integrity Protection may restrict DYLD_LIBRARY_PATH,
    # but PyInstaller also rewrites rpaths, so bundled libs are usually resolved.
    elif sys.platform == 'darwin':
        os.environ['DYLD_LIBRARY_PATH'] = base_path + os.pathsep + os.environ.get('DYLD_LIBRARY_PATH', '')

gdal_data = os.path.join(base_path, 'gdal_data')
proj_data = os.path.join(base_path, 'proj_data')

if os.path.isdir(gdal_data):
    os.environ['GDAL_DATA'] = gdal_data
if os.path.isdir(proj_data):
    os.environ['PROJ_LIB'] = proj_data
