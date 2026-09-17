# vSky2

vSky2 reveals relief variations in Digital Elevation Models (DEMs) through
Volumetric Obscurance (VO) and its Positive / Negative variants (VOP / VON).
It implements the method described in:

Rolland, T.; Monna, F.; Buoncristiani, J.F.; Magail, J.; Esin, Y.; Bohard, B.;
Chateau-Smith, C. *Volumetric Obscurance as a New Tool to Better Visualize
Relief from Digital Elevation Models.* Remote Sens. **2022**, 14, 941.
https://doi.org/10.3390/rs14040941

Ready-to-install packages (Windows MSI, macOS Apple Silicon DMG, QGIS plugin
ZIP) are published on the [Releases](https://github.com/Fabri-Momo/vsky2/releases)
page.

## Installation from source

vSky2 runs on Windows, Linux and macOS with cross-platform GPU acceleration
through [Taichi](https://taichi.graphics/) (NVIDIA CUDA, AMD/Intel Vulkan,
Apple Silicon Metal, or CPU fallback).

```bash
conda env create -f environment.yml
conda activate vsky
```

`environment.yml` includes Taichi. On macOS you can also use
`environment_mac.yml`, which is identical.

If Taichi is not available, vSky2 automatically falls back to NumPy on the CPU.

Then start the program:

```bash
python vSky2.py
```

## Building a standalone application

### Windows

```bash
conda activate vsky
python setup.py
```

The executable is written to `dist/vSky2/vSky2.exe`. To build the MSI installer
(requires the WiX Toolset):

```bash
python build_msi.py
```

### macOS

```bash
conda activate vsky
python -m PyInstaller --noconfirm --clean vSky_mac.spec
```

The bundle is written to `dist/vSky2.app`.

### Linux

```bash
conda activate vsky
python -m PyInstaller --noconfirm --clean vSky_linux.spec
```

## Releases

The version number is defined once in `vsky_version.py` (and mirrored in
`vsky-QGIS_plugin/metadata.txt`). Pushing a tag of the form `vX.Y.Z` triggers
the `Build desktop applications` GitHub Actions workflow, which builds and
smoke-tests the Windows and macOS packages and publishes them as a GitHub
release.

```bash
git tag v2.1.0
git push origin v2.1.0
```

## QGIS plugin

The `vsky-QGIS_plugin` directory contains a Processing plugin compatible with
QGIS 3.34+ (validated on the 3.44 LTR) and QGIS 4.x (Qt 6). See
`vsky-QGIS_plugin/README.md` for details. The plugin is validated on every push
by the `QGIS plugin` workflow.

## Documentation

The in-application documentation lives in `resources/doc` and is embedded
through the Qt resource file `resources.qrc`. Every file must be listed in that
resource file.

After editing `resources.qrc`, regenerate the Python module loaded by the
program:

```bash
pyrcc5 -o qrc_resources.py resources.qrc
```

The documentation is a set of HTML pages. The landing page is `index.html`,
which must link to the other pages; link targets are the aliases declared in
the resource file.

## Language

The user interface is English only. No Qt translator is installed at startup,
so the application looks the same regardless of the system locale.

## License

The license choice is constrained by the components in use:

- PyQt5: https://github.com/PyQt5/PyQt/blob/master/LICENSE (GPL v3.0)
- NumPy, Pillow, GDAL, Taichi
- Python 3
(https://www.gnu.org/licenses/gpl-faq.html#AllCompatibility)
