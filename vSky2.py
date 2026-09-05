import sys, os
import time
import math
from PyQt5 import QtWidgets, QtCore
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
from PyQt5.QtWidgets import (QMainWindow,
        QDialog,
        QApplication, 
        QLabel,
        QLineEdit,
        QComboBox,
        QStackedWidget,
        QFileDialog,
        QAction,
        QCheckBox,
        QVBoxLayout,
        QHBoxLayout,
        QWidget,
        QTextBrowser,
        QPushButton,
        QFormLayout,
        QGroupBox,
        QSpinBox,
        QDoubleSpinBox,
        QMessageBox,
        QProgressBar,
        QPlainTextEdit,
        QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal, QLocale, QTranslator, QObject, QSettings
from osgeo import gdal, osr
gdal.DontUseExceptions()
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
import qrc_resources
qrc_resources.qInitResources()
# Taichi est le moteur de calcul cross-platform (NVIDIA/AMD/Intel/Apple/CPU).
TAICHI_AVAILABLE = False
ti = None
_taichi_accumulate_chunk = None
try:
    import taichi as ti
    if getattr(sys, 'frozen', False):
        import importlib.util
        kernel_path = os.path.join(sys._MEIPASS, 'vsky_taichi.py')
        kernel_spec = importlib.util.spec_from_file_location('vsky_taichi_source', kernel_path)
        kernel_module = importlib.util.module_from_spec(kernel_spec)
        kernel_spec.loader.exec_module(kernel_module)
        _taichi_accumulate_chunk = kernel_module.accumulate_chunk
    else:
        from vsky_taichi import accumulate_chunk as _taichi_accumulate_chunk
    TAICHI_AVAILABLE = True
except Exception:
    TAICHI_AVAILABLE = False
    ti = None

GPU_AVAILABLE = False

# Number of directions processed per Taichi kernel call.
# Larger = fewer kernel launches, but more VRAM. 256 is a good default for Taichi.
DIRECTION_CHUNK_SIZE = 256


def _init_taichi():
    """Initialize Taichi on the main thread. Sets GPU_AVAILABLE."""
    global GPU_AVAILABLE
    if not TAICHI_AVAILABLE:
        return False
    try:
        ti.init(arch=ti.gpu)
        print("[Taichi] GPU backend active")
        GPU_AVAILABLE = True
        return True
    except Exception as e:
        print(f"[Taichi] GPU unavailable ({e}), using CPU")
        ti.init(arch=ti.cpu)
        GPU_AVAILABLE = False
        return True


def _taichi_precompile():
    """Compile the Taichi kernel in the main thread before QThreads use it."""
    if not TAICHI_AVAILABLE:
        return
    _taichi_accumulate_chunk(
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((1, 2), dtype=np.int32),
        np.zeros(1, dtype=np.float32),
        0,
        1,
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
    )
    print("[Taichi] Kernel pre-compiled")


def _compute_taichi(image_final_npy, h_vert_cor, grid, progress_callback=None):
    """Compute VO/VOP accumulators with Taichi. Returns (acc_vo, acc_vop)."""
    H, W = image_final_npy.shape
    acc_vo = np.zeros((H, W), dtype=np.float32)
    acc_vop = np.zeros((H, W), dtype=np.float32)

    image = image_final_npy.astype(np.float32, copy=False)
    h = h_vert_cor.astype(np.float32, copy=False)
    g = grid.astype(np.int32, copy=False)
    n_dirs = g.shape[0]

    chunk_size = DIRECTION_CHUNK_SIZE
    for start in range(0, n_dirs, chunk_size):
        end = min(start + chunk_size, n_dirs)
        _taichi_accumulate_chunk(image, g, h, start, end, acc_vo, acc_vop)
        if progress_callback:
            progress_callback(end - 1)

    return acc_vo, acc_vop


def _compute_numpy(image_final_npy, h_vert_cor, grid, progress_callback=None):
    """Compute VO/VOP accumulators with NumPy (fallback when Taichi is unavailable)."""
    H, W = image_final_npy.shape
    large_vo = np.zeros((H, W), dtype=np.float32)
    large_vop = np.zeros((H, W), dtype=np.float32)

    for i in range(h_vert_cor.size):
        M = np.roll(image_final_npy, [grid[i, 0], grid[i, 1]], axis=[0, 1]) - image_final_npy
        vop = np.clip(h_vert_cor[i] - M, 0, 2 * h_vert_cor[i])
        large_vo += vop
        large_vop += np.minimum(vop, h_vert_cor[i])
        if progress_callback:
            progress_callback(i)

    return large_vo, large_vop


def to_numpy(arr):
    """Compatibility helper: results are already NumPy arrays."""
    return arr


def _local_wkt(name="Local"):
    """Return a minimal LOCAL_CS WKT for ungeoreferenced rasters."""
    return f'LOCAL_CS["{name}"]'


def _open_raster(path):
    """Open a raster with GDAL, falling back to PIL for PNG/JPEG/BMP."""
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found: {}".format(path))

    ds = gdal.Open(path)
    if ds is not None:
        return ds

    try:
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(path)
    except Exception as exc:
        raise IOError(
            "Cannot open {0} as an image: {1}".format(path, exc)
        ) from exc

    if img.mode in ('RGB', 'RGBA', 'P'):
        img = img.convert('L')

    try:
        arr = np.array(img)
    except Exception as exc:
        raise IOError(
            "Cannot read pixel data from {0}: {1}".format(path, exc)
        ) from exc

    if arr.ndim != 2:
        raise ValueError(
            "{0} does not contain a usable 2D grayscale image.".format(path)
        )

    rows, cols = arr.shape
    driver = gdal.GetDriverByName('MEM')
    if driver is None:
        raise RuntimeError("GDAL 'MEM' driver is not available.")
    ds = driver.Create('', cols, rows, 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))
    ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    ds.SetProjection(_local_wkt())
    return ds


class _LogStream(QObject):
    """Thread-safe stdout/stderr redirect to a PyQt signal."""
    new_text = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffer = ""

    def write(self, s):
        if s is None:
            return
        self._buffer += str(s)
        if '\n' in self._buffer:
            lines = self._buffer.split('\n')
            for line in lines[:-1]:
                if line:
                    self.new_text.emit(line)
            self._buffer = lines[-1]
        return len(s)

    def flush(self):
        if self._buffer:
            self.new_text.emit(self._buffer)
            self._buffer = ""


def _gdal_error_handler(err_class, err_num, err_msg):
    """Forward GDAL warnings/errors to the log window."""
    if err_class >= gdal.CE_Warning:
        print("[GDAL] {}".format(err_msg))


class Help(QDialog):

    def __init__(self, page, parent=None):
        super(Help, self).__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setAttribute(Qt.WA_GroupLeader)

        self.pageLabel = QLabel()

        self.textBrowser = QTextBrowser()

        layout = QVBoxLayout()
        layout.addWidget(self.textBrowser)
        self.setLayout(layout)

        self.textBrowser.setSearchPaths([":/"])
        self.textBrowser.setSource(QUrl(page))
        self.resize(400, 600)
        self.setWindowTitle(self.tr("{} Help").format(
                QApplication.applicationName()))


class MainWindow(QMainWindow):
       
    def __init__(self):
        
        super().__init__()
        
        #size of the window
        
        self.title = 'vSky2 - Volumetric Open Sky'
        self.setWindowTitle(self.title)
        base = os.path.dirname(os.path.abspath(__file__))
        if sys.platform == 'win32':
            ico_path = os.path.join(base, 'vSky2.ico')
        elif sys.platform == 'darwin':
            ico_path = os.path.join(base, 'vSky2.icns')
        else:
            ico_path = os.path.join(base, 'vSky2_256x256.png')
        if not os.path.exists(ico_path):
            ico_path = os.path.join(base, 'vSky2_256x256.png')
        self.setWindowIcon(QIcon(ico_path))

        # Persistent settings (last directories, etc.)
        self.settings = QSettings("vSky", "vSky")

        # Enable drag & drop
        self.setAcceptDrops(True)
        
        
        # Image logo at the opening of the program
        
        self.central_widget = QWidget()               
        self.setCentralWidget(self.central_widget)
        
        lay = QVBoxLayout(self.central_widget)
        self._splash_label = QLabel(self)
        self._splash_pixmap = QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vSky2.png'))
        self._splash_label.setPixmap(self._splash_pixmap)
        self._splash_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._splash_label)

        # creation of the menu bar
        
        menubar = self.menuBar()
        file_menu = menubar.addMenu(self.tr('File'))
        info_menu = menubar.addMenu('?')
        
        openAction = QAction(self.tr('Open DEM'), self)  
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip(self.tr("Open a DEM in tif format"))
        openAction.triggered.connect(self.open_image) 
        file_menu.addAction(openAction)

        batchAction = QAction(self.tr('Batch Process'), self)  
        batchAction.setShortcut('Ctrl+B')
        batchAction.setStatusTip(self.tr("Process all DEMs in a folder"))
        batchAction.triggered.connect(self.open_batch) 
        file_menu.addAction(batchAction)

        cutAction = QAction(self.tr('Cut DEM'), self)
        cutAction.setShortcut('Ctrl+Shift+C')
        cutAction.setStatusTip(self.tr("Cut a DEM into georeferenced tiles"))
        cutAction.triggered.connect(self.cut_dem)
        file_menu.addAction(cutAction)

        close_action = QAction(self.tr('Exit'), self)  
        close_action.setShortcut('Ctrl+Q')
        close_action.setStatusTip(self.tr("Exit program"))
        close_action.triggered.connect(self.close) 
        file_menu.addAction(close_action)

        # creation of the tool bar
        toolbar = self.addToolBar(self.tr("Main"))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
                spacing: 4px;
                padding: 4px;
            }
            QToolButton {
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 8px 16px;
                margin: 2px;
                font-weight: 500;
                color: #212529;
            }
            QToolButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QToolButton:pressed {
                background-color: #dee2e6;
                color: #000000;
            }
        """)
        toolbar.addAction(openAction)
        toolbar.addAction(batchAction)
        toolbar.addAction(cutAction)
        toolbar.addAction(close_action)
        
        Info_action = QAction(self.tr('Help'), self)  
        Info_action.triggered.connect(self.click_help) 
        Info_action.setStatusTip(self.tr("General workflow described"))
        info_menu.addAction(Info_action)
        
        about_action = QAction(self.tr('About...'), self)  
        about_action.triggered.connect(self.click_about) 
        about_action.setStatusTip(self.tr("Contributions"))
        info_menu.addAction(about_action)
        
        self.statusBar().showMessage(self.tr('Status Bar'))
        
        self.showMaximized()
        self._rescale_splash()
      
    def _rescale_splash(self):
        if self._splash_label is None or self._splash_pixmap.isNull():
            return
        target_h = max(1, self.height() // 2)
        scaled = self._splash_pixmap.scaledToHeight(target_h, Qt.SmoothTransformation)
        self._splash_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale_splash()

    def open_image(self):
        
        last_dir = self.settings.value("last_open_dir", ".")
        image_path,_= QFileDialog.getOpenFileName(self, self.tr('Open Image'), last_dir,'Image Files(*.tif *.tiff *.png *.jpg *.jpeg);;All Files (*)')
        
        if image_path:      # if open image is canceled return to the main window, otherwise proceed
            self.open_image_path(image_path)
        
    def open_image_path(self, image_path):
        """Open a single DEM (used by menu, drag & drop, recent)."""
        try:
            self.settings.setValue("last_open_dir", os.path.dirname(image_path))
            window = Processing(image_path)
            self.setCentralWidget(window)
            self._splash_label = None
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to open DEM: {}").format(str(e)))

        
    def open_batch(self):
        
        last_dir = self.settings.value("last_batch_dir", ".")
        folder = QFileDialog.getExistingDirectory(self, self.tr('Select folder containing DEMs'), last_dir)
        
        if folder:
            self.settings.setValue("last_batch_dir", folder)
            image_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
            image_files = [f for f in os.listdir(folder) if f.lower().endswith(image_exts)]
            
            if not image_files:
                QMessageBox.warning(self, self.tr("Warning"), self.tr("No DEM files found in the selected folder."))
                return
            
            valid_files = [os.path.join(folder, f) for f in sorted(image_files)]
            
            dialog = BatchDialog(valid_files, self)
            dialog.exec_()

    def cut_dem(self):
        dialog = CutDemDialog(self)
        dialog.exec_()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                self.open_image_path(path)
                break

    def click_help(self):

        form = Help('index.html', self)
        form.show()
    
    def click_about(self):
        msg = QMessageBox(self)
        msg.setWindowTitle(self.tr("About"))
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            "<h2>vSky<sup>2</sup></h2>"
            "<p><b>Volumetric Open Sky</b></p>"
            "<p>For more information contact:<br>"
            "Fabrice Monna: Fabrice.Monna@u-bourgogne.fr<br>"
            "Tanguy Rolland: Tanguy.Rolland@u-bourgogne.fr</p>"
            "<p>Developed with Python 3, PyQt5, NumPy, SciPy, Pillow, GDAL and Taichi.</p>"
            "<p><b>Version 2.0, 2020-2026</b></p>"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        

class CutDemDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Cut DEM"))
        self.setMinimumSize(950, 650)
        self.resize(1150, 750)

        self.im_originale = None
        self.im_array = None
        self.gt = None
        self.proj = None
        self.cols = 0
        self.rows = 0
        self.x_res = 1.0
        self.y_res = 1.0
        self._preview_pixmap = None

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()

        # Left: DEM preview with tiles overlay
        left_box = QGroupBox(self.tr("DEM preview and tiles"))
        left_layout = QVBoxLayout(left_box)
        self.image_label = QLabel(left_box)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222;")
        left_layout.addWidget(self.image_label)
        content_layout.addWidget(left_box, stretch=2)

        # Right: parameters
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        content_layout.addWidget(right_widget, stretch=1)

        main_layout.addLayout(content_layout)

        # Open DEM
        open_btn = QPushButton(self.tr("Open DEM..."), right_widget)
        open_btn.clicked.connect(self._open_dem)
        right_layout.addWidget(open_btn)

        # DEM info
        info_box = QGroupBox(self.tr("DEM information"))
        info_form = QFormLayout(info_box)
        self.lbl_width = QLabel("-")
        self.lbl_height = QLabel("-")
        self.lbl_resolution = QLabel("-")
        self.lbl_proj = QLabel("-")
        self.lbl_epsg = QLabel("-")
        info_form.addRow(self.tr("Width (px / m):"), self.lbl_width)
        info_form.addRow(self.tr("Height (px / m):"), self.lbl_height)
        info_form.addRow(self.tr("Resolution (m/px):"), self.lbl_resolution)
        info_form.addRow(self.tr("Projection:"), self.lbl_proj)
        info_form.addRow(self.tr("EPSG:"), self.lbl_epsg)
        right_layout.addWidget(info_box)

        # Output folder
        out_box = QGroupBox(self.tr("Output folder"))
        out_layout = QHBoxLayout(out_box)
        self.out_edit = QLineEdit(out_box)
        self.out_btn = QPushButton(self.tr("Browse..."), out_box)
        self.out_btn.clicked.connect(self._choose_output)
        out_layout.addWidget(self.out_edit)
        out_layout.addWidget(self.out_btn)
        right_layout.addWidget(out_box)

        # Tile parameters
        tile_box = QGroupBox(self.tr("Tile parameters"))
        self.tile_box = tile_box
        tile_form = QFormLayout(tile_box)

        self.unit_combo = QComboBox(tile_box)
        self.unit_combo.addItem(self.tr("Pixels"), "px")
        self.unit_combo.addItem(self.tr("Meters"), "m")
        self.unit_combo.currentIndexChanged.connect(self._unit_changed)
        self.unit_combo.setMaximumHeight(24)
        tile_form.addRow(self.tr("Unit:"), self.unit_combo)

        self.tile_size_px = QSpinBox(tile_box)
        self.tile_size_px.setRange(0, 100000)
        self.tile_size_px.setSpecialValueText("")
        self.tile_size_px.setValue(0)
        self.tile_size_px.setMaximumHeight(24)
        self.tile_size_px.setKeyboardTracking(False)
        self.tile_size_m = QDoubleSpinBox(tile_box)
        self.tile_size_m.setRange(0.0, 1e9)
        self.tile_size_m.setDecimals(3)
        self.tile_size_m.setSpecialValueText("")
        self.tile_size_m.setValue(0.0)
        self.tile_size_m.setMaximumHeight(24)
        self.tile_size_m.setKeyboardTracking(False)
        self.tile_size_px.valueChanged.connect(self._tile_px_changed)
        self.tile_size_m.valueChanged.connect(self._tile_m_changed)
        self.tile_size_stack = QStackedWidget(tile_box)
        self.tile_size_stack.addWidget(self.tile_size_px)
        self.tile_size_stack.addWidget(self.tile_size_m)
        self.tile_size_stack.setMaximumHeight(24)
        tile_form.addRow(self.tr("Tile size:"), self.tile_size_stack)

        self.overlap_px = QSpinBox(tile_box)
        self.overlap_px.setRange(0, 100000)
        self.overlap_px.setSpecialValueText("")
        self.overlap_px.setValue(0)
        self.overlap_px.setMaximumHeight(24)
        self.overlap_px.setKeyboardTracking(False)
        self.overlap_m = QDoubleSpinBox(tile_box)
        self.overlap_m.setRange(0.0, 1e9)
        self.overlap_m.setDecimals(3)
        self.overlap_m.setSpecialValueText("")
        self.overlap_m.setValue(0.0)
        self.overlap_m.setMaximumHeight(24)
        self.overlap_m.setKeyboardTracking(False)
        self.overlap_px.valueChanged.connect(self._overlap_px_changed)
        self.overlap_m.valueChanged.connect(self._overlap_m_changed)
        self.overlap_stack = QStackedWidget(tile_box)
        self.overlap_stack.addWidget(self.overlap_px)
        self.overlap_stack.addWidget(self.overlap_m)
        self.overlap_stack.setMaximumHeight(24)
        tile_form.addRow(self.tr("Overlap:"), self.overlap_stack)

        self.unit_combo.setCurrentIndex(1)

        right_layout.addWidget(tile_box)
        self.tile_box.setEnabled(False)

        # Action buttons
        self.preview_btn = QPushButton(self.tr("Preview tiles"), right_widget)
        self.preview_btn.clicked.connect(self._preview)
        self.preview_btn.setEnabled(False)
        right_layout.addWidget(self.preview_btn)

        self.cut_btn = QPushButton(self.tr("Cut and save tiles"), right_widget)
        self.cut_btn.clicked.connect(self._cut)
        self.cut_btn.setEnabled(False)
        right_layout.addWidget(self.cut_btn)

        right_layout.addStretch()

        # Path label at the bottom
        bottom_layout = QHBoxLayout()
        self.path_label = QLabel(self.tr("No DEM loaded"))
        self.path_label.setWordWrap(False)
        bottom_layout.addWidget(self.path_label)
        bottom_layout.addStretch()
        main_layout.addLayout(bottom_layout)

    def _open_dem(self):
        last_dir = QSettings("vSky", "vSky").value("last_open_dir", ".")
        image_path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Open DEM"), last_dir,
            self.tr("Image Files (*.tif *.tiff *.png *.jpg *.jpeg);;All Files (*)")
        )
        if not image_path:
            return

        try:
            self.im_originale = _open_raster(image_path)
        except Exception as exc:
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to open DEM: {}").format(exc))
            return

        QSettings("vSky", "vSky").setValue("last_open_dir", os.path.dirname(image_path))

        self.path_label.setText(image_path)

        # Default output folder
        default_out = os.path.join(
            os.path.dirname(image_path),
            os.path.splitext(os.path.basename(image_path))[0] + "_tile"
        )
        self.out_edit.setText(default_out)

        self.cols = self.im_originale.RasterXSize
        self.rows = self.im_originale.RasterYSize

        gt = self.im_originale.GetGeoTransform()
        if gt is None or gt[1] == 0 or gt[5] == 0:
            gt = (0, 1, 0, 0, 0, -1)
        self.gt = gt
        self.x_res = float(gt[1])
        self.y_res = float(abs(gt[5]))

        self.proj = self.im_originale.GetProjection()
        if not self.proj:
            self.proj = _local_wkt()

        raw_array = self.im_originale.GetRasterBand(1).ReadAsArray().astype(np.float32)
        _, self.im_array = tranformImage(raw_array)

        # Build preview pixmap: handle NaN and stretch only the valid range
        im8 = self._build_preview_image(self.im_array)
        im8_pil = Image.fromarray(im8).convert('RGBA')
        im8_data = im8_pil.tobytes('raw', 'RGBA')
        imQt = QImage(im8_data, im8_pil.width, im8_pil.height, QImage.Format_RGBA8888)
        base_pix = QPixmap.fromImage(imQt.copy())

        display_w = 600
        display_h = int(display_w * self.rows / self.cols) if self.cols else 400
        display_h = max(50, min(display_h, 700))
        self._preview_pixmap = base_pix.scaled(display_w, display_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setFixedSize(self._preview_pixmap.width(), self._preview_pixmap.height())
        self.image_label.setPixmap(self._preview_pixmap)

        # Update metadata labels
        width_m = self.cols * self.x_res
        height_m = self.rows * self.y_res
        self.lbl_width.setText("{} px / {:.2f} m".format(self.cols, width_m))
        self.lbl_height.setText("{} px / {:.2f} m".format(self.rows, height_m))
        self.lbl_resolution.setText("{} m/px".format(self.x_res))

        srs = osr.SpatialReference(wkt=self.proj)
        projection = srs.GetAttrValue('projcs')
        if projection is None:
            projection = self.tr("Local")
        epsg = srs.GetAttrValue('AUTHORITY',1)
        if epsg is None:
            epsg = self.tr("N/A")
        self.lbl_proj.setText(projection)
        self.lbl_epsg.setText(epsg)

        # Update spinbox ranges and defaults
        width_m = self.cols * self.x_res
        height_m = self.rows * self.y_res
        max_m = max(width_m, height_m)

        tile_m = _round_to_first_sig(max_m / 8.0)
        tile_px = max(1, min(self.cols, int(round(tile_m / self.x_res))))
        self.tile_size_px.setRange(1, self.cols)
        self.tile_size_px.setValue(tile_px)

        overlap_m = _round_to_first_sig(tile_m / 8.0)
        overlap_px = max(0, min(tile_px - 1, int(round(overlap_m / self.x_res))))
        self.overlap_px.setSpecialValueText("0")
        self.overlap_m.setSpecialValueText("0")
        self.overlap_px.setValue(overlap_px)

        self._sync_m_from_px()

        if self.x_res > 0:
            decs = max(3, 1 + int(math.ceil(-math.log10(self.x_res))))
            self.tile_size_m.setDecimals(decs)
            self.overlap_m.setDecimals(decs)
            self.tile_size_m.setSingleStep(self.x_res)
            self.overlap_m.setSingleStep(self.x_res)

        self.tile_box.setEnabled(True)
        for w in (self.tile_size_px, self.tile_size_m, self.overlap_px, self.overlap_m):
            w.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.cut_btn.setEnabled(True)

    def _build_preview_image(self, image_input):
        return array_image(image_input, 2.5, 97.5)

    def _choose_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Select output folder"), self.out_edit.text() or "."
        )
        if folder:
            self.out_edit.setText(folder)

    def _sync_m_from_px(self):
        x_res = self.x_res if self.x_res else 1.0

        self._update_overlap_max()

        self.tile_size_m.blockSignals(True)
        self.tile_size_m.setValue(self.tile_size_px.value() * x_res)
        self.tile_size_m.blockSignals(False)

        self.overlap_m.blockSignals(True)
        self.overlap_m.setValue(self.overlap_px.value() * x_res)
        self.overlap_m.blockSignals(False)

    def _update_overlap_max(self):
        x_res = self.x_res if self.x_res else 1.0
        tile_px = self.tile_size_px.value()
        max_px = max(0, tile_px - 1)
        max_m = max(0.0, max_px * x_res)
        self.overlap_px.setMaximum(max_px)
        self.overlap_m.setMaximum(max_m)

    def _unit_changed(self, index):
        self.tile_size_stack.setCurrentIndex(index)
        self.overlap_stack.setCurrentIndex(index)
        self._sync_m_from_px()

        current_tile = self.tile_size_stack.currentWidget()
        if current_tile:
            current_tile.setEnabled(True)
        current_overlap = self.overlap_stack.currentWidget()
        if current_overlap:
            current_overlap.setEnabled(True)

    def _tile_px_changed(self, value):
        self._sync_m_from_px()

    def _tile_m_changed(self, value):
        x_res = self.x_res if self.x_res else 1.0
        self.tile_size_px.blockSignals(True)
        self.tile_size_px.setValue(int(round(value / x_res)))
        self.tile_size_px.blockSignals(False)
        self._sync_m_from_px()

    def _overlap_px_changed(self, value):
        x_res = self.x_res if self.x_res else 1.0
        self.overlap_m.blockSignals(True)
        self.overlap_m.setValue(value * x_res)
        self.overlap_m.blockSignals(False)

    def _overlap_m_changed(self, value):
        x_res = self.x_res if self.x_res else 1.0
        self.overlap_px.blockSignals(True)
        self.overlap_px.setValue(int(round(value / x_res)))
        self.overlap_px.blockSignals(False)

    def _preview(self):
        if self._preview_pixmap is None:
            return
        pixmap = self._preview_pixmap.copy()
        painter = QPainter(pixmap)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)

        tile = self.tile_size_px.value()
        ov = self.overlap_px.value()
        if ov >= tile:
            painter.end()
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Overlap must be smaller than tile size."))
            return

        scale_x = pixmap.width() / self.cols
        scale_y = pixmap.height() / self.rows

        for y in range(0, self.rows, max(1, tile - ov)):
            y2 = min(y + tile, self.rows)
            for x in range(0, self.cols, max(1, tile - ov)):
                x2 = min(x + tile, self.cols)
                rx = int(x * scale_x)
                ry = int(y * scale_y)
                rw = max(1, int((x2 - x) * scale_x))
                rh = max(1, int((y2 - y) * scale_y))
                painter.drawRect(rx, ry, rw, rh)

        painter.end()
        self.image_label.setPixmap(pixmap)

    def _cut(self):
        if self.im_array is None:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Open a DEM first."))
            return

        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Select an output folder."))
            return
        if not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as exc:
                QMessageBox.warning(self, self.tr("Warning"), self.tr("Could not create output folder: {}").format(exc))
                return

        tile = self.tile_size_px.value()
        ov = self.overlap_px.value()
        if ov >= tile:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Overlap must be smaller than tile size."))
            return
        step = max(1, tile - ov)

        driver = gdal.GetDriverByName('GTIFF')
        desc = self.im_originale.GetDescription() or ""
        basename = os.path.splitext(os.path.basename(desc))[0] or "dem"

        count = 0
        for row, y in enumerate(range(0, self.rows, step)):
            y2 = min(y + tile, self.rows)
            for col, x in enumerate(range(0, self.cols, step)):
                x2 = min(x + tile, self.cols)
                if x2 <= x or y2 <= y:
                    continue

                tile_arr = self.im_array[y:y2, x:x2]
                if np.all(np.isnan(tile_arr)):
                    continue

                x_geo = self.gt[0] + x * self.gt[1] + y * self.gt[2]
                y_geo = self.gt[3] + x * self.gt[4] + y * self.gt[5]
                new_gt = (x_geo, self.gt[1], self.gt[2], y_geo, self.gt[4], self.gt[5])

                outname = os.path.join(out_dir, "{}_tile_{:02d}_{:02d}.tif".format(basename, row, col))
                out_ds = driver.Create(outname, x2 - x, y2 - y, 1, gdal.GDT_Float32)
                if out_ds is None:
                    continue

                out_ds.SetGeoTransform(new_gt)
                out_ds.SetProjection(self.proj)
                band = out_ds.GetRasterBand(1)
                band.WriteArray(tile_arr)
                out_ds.FlushCache()
                out_ds = None
                count += 1

        QMessageBox.information(self, self.tr("Done"), self.tr("Saved {} tile(s) to {}.").format(count, out_dir))


class AspectRatioPixmapLabel(QLabel):
    def __init__(self, parent):
        super(AspectRatioPixmapLabel, self).__init__(parent)
        self.pix = None
        self.setMinimumSize(1, 1)
        self.setScaledContents(False)

    def setPixmap(self, pixmap):
        self.pix = pixmap
        super().setPixmap(self.scaledPixmap())

    def heightForWidth(self, width):
        return self.height() if self.pix else self.pix.height() * width / self.pix.width()

    def sizeHint(self):
        width = self.width()
        return QtCore.QSize(width, self.heightForWidth(width))

    def scaledPixmap(self):
        scaled = self.pix.scaled(self.size() * self.devicePixelRatioF(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled.setDevicePixelRatio(self.devicePixelRatioF())
        return scaled

    def resizeEvent(self, event):
        if self.pix:
            super().setPixmap(self.scaledPixmap())


class Processing(QWidget):
      
    def __init__(self, image_path):
        
        super().__init__()
        self.prep_blur_radius = 2
        self.exageration_value = 1
        self.checked_vo = True
        self.checked_vop = False
        self.checked_von = False
        self.image_folder, self.image_name = os.path.split(image_path)
        self.default_output_folder = os.path.join(
            self.image_folder, os.path.splitext(self.image_name)[0] + "_processing"
        )
        self.setup_ui()
        self._log_stream = _LogStream(self)
        self._log_stream.new_text.connect(self.log_message)
        sys.stdout = self._log_stream
        sys.stderr = self._log_stream
        self.extract_image(image_path)
        self.log_message("[Taichi] Backend: {}".format("GPU" if GPU_AVAILABLE else "CPU"))

    def setup_ui(self):
        # create and set layout to place widgets
        primary_layout = QHBoxLayout(self)
        self.setLayout(primary_layout)

        image_box = QGroupBox(self)
        secondary_left_layout = QVBoxLayout(image_box)
        image_box.setLayout(secondary_left_layout)

        self.image = AspectRatioPixmapLabel(image_box)
        secondary_left_layout.addWidget(self.image, stretch=3)

        image_location = QWidget(image_box)
        image_location_layout = QFormLayout()
        image_location.setLayout(image_location_layout)
        image_location_label = QLabel()
        image_location_label.setText(self.tr('Path of the DEM:'))
        self.image_location_value = QLabel()
        self.image_location_value.setText(os.path.join(self.image_folder, self.image_name))
        image_location_layout.addRow(image_location_label, self.image_location_value)
        secondary_left_layout.addWidget(image_location)

        right_block = QWidget(self)
        right_block.setMinimumSize(300,600)
        secondary_right_layout = QVBoxLayout(right_block)
        primary_layout.addWidget(image_box, stretch=2)
        primary_layout.addWidget(right_block, stretch=1)

        data_box = QGroupBox(self.tr('DEM parameters'))
        data_form = QFormLayout(data_box)
        data_box.setLayout(data_form)
        
        data_proj_label = QLabel(self)
        data_proj_label.setText(self.tr('Projection:'))       
        self.data_proj_value = QLabel(self)       
        data_form.addRow(data_proj_label, self.data_proj_value)

        data_epsg_label = QLabel(self)
        data_epsg_label.setText(self.tr('EPSG:'))       
        self.data_epsg_value = QLabel(self)       
        data_form.addRow(data_epsg_label, self.data_epsg_value)

        data_res_label = QLabel(self)
        data_res_label.setText(self.tr('Resolution (m/px):'))       
        self.data_res_value = QLabel(self)       
        data_form.addRow(data_res_label, self.data_res_value)
        
        data_height_label = QLabel(self)
        data_height_label.setText(self.tr('Height (px / m):'))       
        self.data_height_value = QLabel(self)       
        data_form.addRow(data_height_label, self.data_height_value)
        
        data_width_label = QLabel(self)
        data_width_label.setText(self.tr('Width (px / m):'))       
        self.data_width_value = QLabel(self)
        data_form.addRow(data_width_label, self.data_width_value)
        

        secondary_right_layout.addWidget(data_box)

        prep_box = QGroupBox(self.tr('Preparation of the DEM'))
        #prep_box.setFrameStyle(QFrame.Panel | QFrame.Plain)
        prep_box_layout = QFormLayout()

        self.check_blur = QCheckBox(self.tr("Prior smoothing (Gaussian kernel)"),self)
        self.check_blur.setChecked(False)
        self.check_blur.toggled.connect(self.toggle_preprocessing)
        self.prep_blur_label = QLabel(prep_box)
        self.prep_blur_label.setText(self.tr('Radius (in pixels):'))
        self.prep_blur_label.setEnabled(False)
        self.prep_blur_spin = QSpinBox(self)
        self.prep_blur_spin.setRange(1,100)
        self.prep_blur_spin.setValue(self.prep_blur_radius)
        self.prep_blur_spin.setEnabled(False)                    

        prep_box_layout.addRow(self.check_blur)  
        prep_box_layout.addRow(self.prep_blur_label,self.prep_blur_spin) 
        prep_box.setLayout(prep_box_layout)

        calc_param_box = QGroupBox(self.tr('Calculation parameters'))
        calc_param_layout = QFormLayout()
        calc_param_box.setLayout(calc_param_layout)

        self.calc_param_radius_unit = QComboBox(calc_param_box)
        self.calc_param_radius_unit.addItem(self.tr("Pixels"), "px")
        self.calc_param_radius_unit.addItem(self.tr("Meters"), "m")
        self.calc_param_radius_unit.currentIndexChanged.connect(self._radius_unit_changed)
        self.calc_param_radius_unit.setMaximumWidth(100)
        calc_param_layout.addRow(self.tr("Radius unit:"), self.calc_param_radius_unit)

        self.calc_param_radius_spin = QSpinBox(calc_param_box)
        self.calc_param_radius_spin.setRange(2, 100)
        self.calc_param_radius_spin.setValue(5)
        self.calc_param_radius_spin.setKeyboardTracking(False)
        self.calc_param_radius_spin.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.calc_param_radius_m = QDoubleSpinBox(calc_param_box)
        self.calc_param_radius_m.setRange(0.0, 1e9)
        self.calc_param_radius_m.setDecimals(3)
        self.calc_param_radius_m.setSpecialValueText("")
        self.calc_param_radius_m.setValue(0.0)
        self.calc_param_radius_m.setKeyboardTracking(False)
        self.calc_param_radius_m.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.calc_param_radius_spin.valueChanged.connect(self._radius_px_changed)
        self.calc_param_radius_m.valueChanged.connect(self._radius_m_changed)
        self.calc_param_radius_stack = QStackedWidget(calc_param_box)
        self.calc_param_radius_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.calc_param_radius_stack.addWidget(self.calc_param_radius_spin)
        self.calc_param_radius_stack.addWidget(self.calc_param_radius_m)
        calc_param_layout.addRow(self.tr("Radius:"), self.calc_param_radius_stack)

        calc_param_exageration_label = QLabel(self)
        calc_param_exageration_label.setText(self.tr('z exageration:'))
        self.calc_param_exageration = QSpinBox(self)
        self.calc_param_exageration.setRange(1,1000000)
        self.calc_param_exageration.setValue(self.exageration_value)
        calc_param_layout.addRow(calc_param_exageration_label, self.calc_param_exageration)
        
        calc_param_method = QWidget(calc_param_box)
        calc_param_method_layout = QHBoxLayout()
        calc_param_method.setLayout(calc_param_method_layout)
        self.calc_param_vo = QCheckBox(self.tr("VO"), self)
        self.calc_param_vo.setChecked(self.checked_vo)
        self.calc_param_vo.setEnabled(False)
        self.calc_param_vo.setToolTip (self.tr('VO is systematically computed'))       
        self.calc_param_vop = QCheckBox(self.tr("VOP"), self)
        self.calc_param_vop.setChecked(self.checked_vop)
        self.calc_param_von = QCheckBox(self.tr("VON"), self)
        self.calc_param_von.setChecked(self.checked_von)
        calc_param_method_layout.addWidget(self.calc_param_vo)
        calc_param_method_layout.addWidget(self.calc_param_vop)
        calc_param_method_layout.addWidget(self.calc_param_von)
        calc_param_method_layout.addStretch()
        calc_param_layout.addRow(calc_param_method)

        output_box = QGroupBox(self.tr('Output parameters'))
        output_layout = QFormLayout()
        output_box.setLayout(output_layout)
        #output_box.setFrameStyle(QFrame.Panel | QFrame.Plain)
        
        self.check_8_bits = QCheckBox(self.tr("8-bits output"),output_box)
        self.check_8_bits.setChecked(True)
        output_layout.addRow(self.check_8_bits)

        output_folder_box = QGroupBox(self.tr("Output folder"))
        output_folder_layout = QHBoxLayout(output_folder_box)
        self.output_folder_edit = QLineEdit(self.default_output_folder, output_folder_box)
        self.output_folder_btn = QPushButton(self.tr("Browse..."), output_folder_box)
        self.output_folder_btn.clicked.connect(self._choose_output_folder)
        output_folder_layout.addWidget(self.output_folder_edit)
        output_folder_layout.addWidget(self.output_folder_btn)

        launch_box = QGroupBox(self.tr('Launch calculation'))
        launch_layout = QVBoxLayout()
        launch_box.setLayout(launch_layout)
        #launch_box.setFrameStyle(QFrame.Panel | QFrame.Plain)
        self.bt_calc = QPushButton(self.tr("Calculate"), launch_box)
        self.bt_calc.setFixedWidth(100)
        self.bt_calc.setToolTip(self.tr("Run the calculation <i>Calculate</i>"))      
        self.bt_calc.clicked.connect(self.click_calc)
        launch_layout.addWidget(self.bt_calc)
        self.toggle_calc_mode = QPushButton(self.tr("Use GPU"), launch_box)
        self.toggle_calc_mode.setCheckable(True)
        self.toggle_calc_mode.setEnabled(GPU_AVAILABLE)
        self.toggle_calc_mode.setChecked(GPU_AVAILABLE)
        launch_layout.addWidget(self.toggle_calc_mode)
        self.progress = QProgressBar(launch_box)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMaximum(100)
        launch_layout.addWidget(self.progress)

        self.log_text = QPlainTextEdit(launch_box)
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(200)
        self.log_text.setMaximumHeight(120)
        self.log_text.setMinimumHeight(60)
        launch_layout.addWidget(self.log_text)

        secondary_right_layout.addWidget(prep_box)
        secondary_right_layout.addWidget(calc_param_box)
        secondary_right_layout.addWidget(output_box)
        secondary_right_layout.addWidget(output_folder_box)
        secondary_right_layout.addStretch()
        secondary_right_layout.addWidget(launch_box)

    def _choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Select output folder"), self.output_folder_edit.text() or self.image_folder
        )
        if folder:
            self.output_folder_edit.setText(folder)

    def log_message(self, text):
        if hasattr(self, 'log_text'):
            self.log_text.appendPlainText(text)

    def extract_image(self, image_path):
              
        # destroy result windows if they exist
        if hasattr(self, 'win0'):
            self.win0.close()
        
        if hasattr(self, 'win1'):
            self.win1.close()
            
        if hasattr(self, 'win2'):
            self.win2.close()

        if hasattr(self, 'win3'):
            self.win3.close()            
        
        # extraction of name of the file, georeferencement, band…
        
        self.im_originale = _open_raster(image_path)
        self.cols = self.im_originale.RasterXSize
        self.rows = self.im_originale.RasterYSize                            
        gt = self.im_originale.GetGeoTransform()
        if gt is None or gt[1] == 0 or gt[5] == 0:
            gt = (0, 1, 0, 0, 0, -1)
        self.gt = gt
        _, self.x_res, _, _, _, self.y_res = gt
        
        if self.x_res == 0:
            self.x_res = 1.0
        if self.y_res == 0:
            self.y_res = -1.0
        self.y_res = float(abs(self.y_res))

        proj = self.im_originale.GetProjection()
        if not proj:
            proj = _local_wkt()
        self.proj = proj
        srs = osr.SpatialReference(wkt=proj)
        projection = srs.GetAttrValue('projcs')
        if projection is None:
            projection = self.tr("Local")
        self.epsg = srs.GetAttrValue('AUTHORITY',1)
        if self.epsg is None:
            self.epsg = self.tr("N/A")

        # transformation as an image, extraction of size, elimination of nan, contrast improvement
        
        im_numpy = self.im_originale.GetRasterBand(1).ReadAsArray()                                              
        self.wherenan, self.image_final_npy = tranformImage(im_numpy)
        image8bits = array_image(self.image_final_npy, 2.5, 97.5)
        
        im8 = Image.fromarray(image8bits).convert('RGBA')
        im8_data = im8.tobytes('raw', 'RGBA')
        imQt = QImage(im8_data, im8.width, im8.height, QImage.Format_RGBA8888)
        self.original_pixmap = QPixmap.fromImage(imQt.copy())

        self.image.setPixmap(self.original_pixmap)

        # Replacement of values (by convention) by Nan
        
        self.image_final_npy = np.nan_to_num(self.image_final_npy)
        
        # layout construction
        
        self.data_proj_value.setText(projection)
        self.data_epsg_value.setText(self.epsg)
        width_m = self.cols * self.x_res
        height_m = self.rows * self.y_res
        self.data_res_value.setText(f'{self.x_res}')
        self.data_height_value.setText("{} px / {:.2f} m".format(self.rows, height_m))
        self.data_width_value.setText("{} px / {:.2f} m".format(self.cols, width_m))

        self._update_radius_m_from_res()

    def _radius_unit_changed(self, index):
        if index == 0:
            self.calc_param_radius_spin.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.calc_param_radius_m.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        else:
            self.calc_param_radius_m.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.calc_param_radius_spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.calc_param_radius_stack.setCurrentIndex(index)
        self.calc_param_radius_stack.updateGeometry()
        self._sync_radius_m_from_px()

    def _radius_px_changed(self, value):
        self._sync_radius_m_from_px()

    def _radius_m_changed(self, value):
        x_res = getattr(self, 'x_res', None)
        if not x_res or x_res <= 0:
            return
        px = max(2, int(round(value / x_res)))
        max_px = self.calc_param_radius_spin.maximum()
        px = min(px, max_px)
        self.calc_param_radius_spin.blockSignals(True)
        self.calc_param_radius_spin.setValue(px)
        self.calc_param_radius_spin.blockSignals(False)
        self._sync_radius_m_from_px()

    def _sync_radius_m_from_px(self):
        x_res = getattr(self, 'x_res', None)
        if not x_res or x_res <= 0:
            return
        m_value = self.calc_param_radius_spin.value() * x_res
        self.calc_param_radius_m.blockSignals(True)
        self.calc_param_radius_m.setValue(m_value)
        self.calc_param_radius_m.blockSignals(False)

    def _update_radius_m_from_res(self):
        x_res = getattr(self, 'x_res', None)
        if not x_res or x_res <= 0:
            return
        px_max = self.calc_param_radius_spin.maximum()
        self.calc_param_radius_m.setRange(2 * x_res, px_max * x_res)
        decs = max(3, 1 + int(math.ceil(-math.log10(x_res))))
        self.calc_param_radius_m.setDecimals(decs)
        self.calc_param_radius_m.setSingleStep(x_res)
        self._sync_radius_m_from_px()

    def toggle_preprocessing(self):
        
        "Set variables and states for preprocessing"
        if self.check_blur.isChecked():
            self.prep_blur_spin.setEnabled(True)
            self.prep_blur_label.setEnabled(True)
        else:
            self.prep_blur_spin.setEnabled(False)
            self.prep_blur_label.setEnabled(False)

    def click_calc(self):
        output_folder = self.output_folder_edit.text().strip()
        if not output_folder:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Select an output folder."))
            return
        output_folder = os.path.abspath(os.path.expanduser(output_folder))
        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("Could not create output folder: {}").format(exc)
            )
            return
        self.output_folder_edit.setText(output_folder)
                     
#         close result windows if they already exist  
               
        if hasattr(self, 'win0'):
            self.win0.close()
        
        if hasattr(self, 'win1'):
            self.win1.close()
            
        if hasattr(self, 'win2'):
            self.win2.close()
            
        if hasattr(self, 'win3'):
            self.win3.close()            
        
        # contruction of a grid of pixels around the point of interest (i.e. grid)
        radius_value = self.calc_param_radius_spin.value()
        exageration_value = self.calc_param_exageration.value()
        x_res = float(getattr(self, 'x_res', 0.0))
        cache_key = (radius_value, exageration_value, round(x_res, 6))

        if not hasattr(self, '_calc_cache'):
            self._calc_cache = {}

        if cache_key in self._calc_cache:
            self.grid, self.d, self.h_vert_cor = self._calc_cache[cache_key]
        else:
            x = np.arange(-radius_value, radius_value + 1, dtype=int)
            y = np.arange(-radius_value, radius_value + 1, dtype=int)
            grid = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)
            distance = np.sqrt(grid[:, 0]**2 + grid[:, 1]**2)
            self.d = distance[distance < radius_value]
            self.grid = grid[distance < radius_value]
            self.h_vert_cor = np.sqrt(radius_value**2 - self.d**2) * self.x_res / exageration_value
            self._calc_cache[cache_key] = (self.grid, self.d, self.h_vert_cor)      
        
        # define max for progression bar
        
        self.progress.setMaximum(self.d.size-1)

        checked_vop = self.calc_param_vop.isChecked()
        checked_von = self.calc_param_von.isChecked()
        checked_cuda = self.toggle_calc_mode.isChecked()
        # calculation 
        if self.check_blur.isChecked():    # blurring option 
            self.calc = Calculation(self.grid, self.d, self.h_vert_cor,
                    blurring(self.image_final_npy, self.prep_blur_spin.value()), checked_vop, checked_von, checked_cuda)
            
        else:                      #no blur
            self.calc = Calculation(self.grid, self.d, self.h_vert_cor,
                            self.image_final_npy, checked_vop, checked_von, checked_cuda)        
            
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.progressChanged.connect(self.progress.setValue)
        self.calc.start()       
        
        
    def onCountChanged(self, value):
            
        self.progress.setValue(value)
        
        #print(self.d.size, self.progress.value())        
        
        # when calculation is done, save and show results
        # Use self.sender() to get the actual Calculation instance, because
        # self.calc may have been overwritten if the user started a new run.
        
        if self.progress.value() == (self.d.size -1):
            
            calc = self.sender()
            
            shift = 0
            self.progress.setValue(0) # set the progression bar to zero
            
            self.out_vo = to_numpy(calc.large_vo[:,:,0]) / (2 * self.h_vert_cor.sum())  # collect result for VO
            self.out_vo[self.wherenan] = np.nan 
            self.nm = 'VO'       
            self.im = self.out_vo
            self.saveImage()  
            self.win0 = PopupWin(self.im, shift, self.newname)    
            self.win0.show()
            
            if self.calc_param_vop.isChecked():
                
                shift = shift + 50
                self.out_vop = to_numpy(calc.large_vop[:,:,0]) / self.h_vert_cor.sum()    # result for vop
                self.out_vop[self.wherenan] = np.nan 
                self.nm = 'VOP'
                self.im = self.out_vop
                self.saveImage()                   
                self.win1 = PopupWin(self.im, shift, self.newname)  
                self.win1.show()
                
            if self.calc_param_von.isChecked():
                
                shift = shift + 50
                self.out_von = to_numpy(calc.large_von[:,:,0]) / self.h_vert_cor.sum()   # result for von
                self.out_von[self.wherenan] = np.nan 
                self.nm = 'VON'
                self.im = self.out_von
                self.saveImage()
                self.win2 = PopupWin(self.im, shift, self.newname)        
                self.win2.show()

            if self.calc_param_von.isChecked() and self.calc_param_vop.isChecked():
                
                shift = shift + 50
                self.nm = 'RGB_combine'
                
                self.out_vo = to_numpy(self.out_vo)
                self.out_vop = to_numpy(self.out_vop)
                self.out_von = to_numpy(self.out_von)
                    
                self.saveImageRGB()
                self.win3 = PopupWinRGB(self.out_image, shift, self.newname)        
                self.win3.show()

    def _build_output_name(self, extension='tif'):
        
        ###### construction of the name of the file
        
        exageration_value = self.calc_param_exageration.value()
        radius_value = self.calc_param_radius_spin.value()
        blur_value = self.prep_blur_spin.value()
        if exageration_value == 1:        
            exagere = ''
        else:
            exagere = '_Z-exag='+str(exageration_value)
        
        if self.check_blur.isChecked():
            blur = '_smooth_r='+str(blur_value)
        else:
            blur = ''

        basename = os.path.splitext(self.image_name)[0]
        output_folder = self.output_folder_edit.text().strip() or self.default_output_folder
        name = os.path.join(output_folder, f'{basename}_{self.nm}_r={radius_value}{blur}{exagere}.{extension}')
        return name

    def saveImage(self):            
        
        ###### write results  

        self.newname = self._build_output_name('tif')

        # Save as geotif
        
        cols = self.im_originale.RasterXSize
        rows = self.im_originale.RasterYSize
        bands = 1
        gt = self.gt
        proj = self.proj

        driver = gdal.GetDriverByName('GTIFF')    
        driver.Register()

        output = driver.Create(self.newname, cols, rows, bands, gdal.GDT_Float32)
        output.SetGeoTransform(gt)
        output.SetProjection(proj)
        outBand = output.GetRasterBand(1)
        outBand.WriteArray(self.im, 0, 0)
        output.FlushCache()
        output = None
        outBand = None
        self.log_message(self.tr("Processed file saved: {}").format(self.newname))
        
        # Construction of visual outputs
        
        self.out_image = array_image(self.im,2.5,97.5)
        self.out_image = self.out_image.astype(np.uint8)
               
        # possible save as 8bits jpg
        
        if self.check_8_bits.isChecked():
            newname8bits = self._build_output_name('jpg').replace(f'_{self.nm}_', f'_{self.nm}_8bits_')
            self.im = Image.fromarray(self.out_image)
            self.im.save(newname8bits)
            self.log_message(self.tr("Processed file saved: {}").format(newname8bits))

    def saveImageRGB(self):            
        
        ###### write results  

        self.newname = self._build_output_name('tif')
        
        # Construction of visual outputs  
        
        r_channel = array_image(self.out_vo,2.5,97.5)
        g_channel = array_image(self.out_vop,2.5,97.5)
        b_channel = array_image(self.out_von,2.5,97.5)
        self.out_image = np.dstack((r_channel, g_channel, b_channel))
        self.out_image = self.out_image.astype(np.uint8)
        
        #  save as 8bits RGB jpg
        if self.check_8_bits.isChecked():
            newname8bits = self._build_output_name('jpg').replace(f'_{self.nm}_', f'_{self.nm}_8bits_')
            im = Image.fromarray(self.out_image, mode='RGB')
            im.save(newname8bits)
            self.log_message(self.tr("Processed file saved: {}").format(newname8bits))
            
                  
class Calculation(QThread):

    
    countChanged = pyqtSignal(int)
    progressChanged = pyqtSignal(int)
    collect = pyqtSignal()

    
    def __init__(self, grid, d, h_vert_cor,
                 image_final_npy, checked_vop, checked_von, checked_cuda):
        
        super(Calculation, self).__init__()
        
        self.h_vert_cor = h_vert_cor
        self.d = d
        self.grid = grid
        self.image_final_npy = image_final_npy
        self.checked_von = checked_von
        self.checked_vop = checked_vop
        self.use_cuda = checked_cuda

        
    def run(self):
        # calculation of VO, VOP, VON

        use_gpu = GPU_AVAILABLE and self.use_cuda
        start_time = time.time()

        def _emit_progress(i):
            self.progressChanged.emit(i)

        if use_gpu and TAICHI_AVAILABLE:
            acc_vo, acc_vop = _compute_taichi(
                self.image_final_npy, self.h_vert_cor, self.grid,
                progress_callback=_emit_progress
            )
        else:
            acc_vo, acc_vop = _compute_numpy(
                self.image_final_npy, self.h_vert_cor, self.grid,
                progress_callback=_emit_progress
            )

        sum_h = float(self.h_vert_cor.sum())
        H, W = acc_vo.shape
        self.large_vo = np.zeros((H, W, 2), dtype=np.float32)
        self.large_vo[:, :, 0] = acc_vo

        if self.checked_vop or self.checked_von:
            self.large_vop = np.zeros((H, W, 2), dtype=np.float32)
            self.large_vop[:, :, 0] = acc_vop

        if self.checked_von:
            self.large_von = np.zeros((H, W, 2), dtype=np.float32)
            self.large_von[:, :, 0] = -(acc_vo - acc_vop - sum_h)

        print(self.tr("Temps d execution : {} secondes ---").format(time.time() - start_time))
        self.countChanged.emit(self.d.size - 1)
        self.quit()
           
class PopupWin(QWidget):
    
    def __init__(self, im, shift, name):
        
        super(PopupWin, self).__init__()
        
        # popup result window (900x900 in size)
        
        Win = 900
        
        image8bits = array_image(im, 2.5,97.5)
        im8 = Image.fromarray(image8bits).convert('RGBA')
        im8_data = im8.tobytes('raw', 'RGBA')
        imQt = QImage(im8_data, im8.width, im8.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(imQt.copy())
        pixmap = pixmap.scaled(Win, Win, Qt.KeepAspectRatio)  
                
        self.setWindowTitle(name)
        self.setFixedSize(Win, Win)
        
        layout = QHBoxLayout()
        label_image = QLabel(self)
        label_image.setPixmap(pixmap)
        label_image.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_image)
        self.setLayout(layout)
        self.move(shift, shift)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)


class PopupWinRGB(QWidget):
    
    def __init__(self, im, shift, name):
        
        super(PopupWinRGB, self).__init__()
        
        # popup result window (900x900 in size)
        
        Win = 900
        
        height, width, channel = im.shape
        bytesPerLine = 3 * width
        qImg = QImage(im.data, width, height, bytesPerLine, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qImg)
        pixmap = pixmap.scaled(Win, Win, Qt.KeepAspectRatio)  
                
        self.setWindowTitle(name)
        self.setFixedSize(Win, Win)
        
        layout = QHBoxLayout()
        label_image = QLabel(self)
        label_image.setPixmap(pixmap)
        label_image.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_image)
        self.setLayout(layout)
        self.move(shift, shift)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        

class BatchDialog(QDialog):

    def __init__(self, file_list, parent=None):
        super(BatchDialog, self).__init__(parent)
        self.file_list = file_list
        self.setWindowTitle(self.tr("Batch Processing"))
        self.setMinimumWidth(500)
        self.x_res = self._get_first_x_res()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # File info
        info_box = QGroupBox(self.tr("Files"))
        info_layout = QFormLayout(info_box)
        info_layout.addRow(self.tr("Folder:"), QLabel(os.path.dirname(self.file_list[0])))
        info_layout.addRow(self.tr("DEMs found:"), QLabel(str(len(self.file_list))))
        layout.addWidget(info_box)

        # Parameters
        param_box = QGroupBox(self.tr("Calculation parameters"))
        param_layout = QFormLayout(param_box)

        self.radius_unit = QComboBox(param_box)
        self.radius_unit.addItem(self.tr("Pixels"), "px")
        self.radius_unit.addItem(self.tr("Meters"), "m")
        self.radius_unit.currentIndexChanged.connect(self._radius_unit_changed)
        self.radius_unit.setMaximumWidth(100)
        param_layout.addRow(self.tr("Radius unit:"), self.radius_unit)

        self.radius_spin = QSpinBox(param_box)
        self.radius_spin.setRange(2, 100)
        self.radius_spin.setValue(5)
        self.radius_spin.setKeyboardTracking(False)
        self.radius_spin.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.radius_m = QDoubleSpinBox(param_box)
        self.radius_m.setRange(0.0, 1e9)
        self.radius_m.setDecimals(3)
        self.radius_m.setSpecialValueText("")
        self.radius_m.setValue(0.0)
        self.radius_m.setKeyboardTracking(False)
        self.radius_m.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.radius_spin.valueChanged.connect(self._radius_px_changed)
        self.radius_m.valueChanged.connect(self._radius_m_changed)
        self.radius_stack = QStackedWidget(param_box)
        self.radius_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.radius_stack.addWidget(self.radius_spin)
        self.radius_stack.addWidget(self.radius_m)
        param_layout.addRow(self.tr("Radius:"), self.radius_stack)

        self.exageration_spin = QSpinBox(self)
        self.exageration_spin.setRange(1, 1000000)
        self.exageration_spin.setValue(1)
        param_layout.addRow(self.tr("z exageration:"), self.exageration_spin)

        self.check_blur = QCheckBox(self.tr("Prior smoothing (Gaussian kernel)"), self)
        self.check_blur.setChecked(False)
        self.check_blur.toggled.connect(self.toggle_blur)
        param_layout.addRow(self.check_blur)

        self.blur_spin = QSpinBox(self)
        self.blur_spin.setRange(1, 100)
        self.blur_spin.setValue(2)
        self.blur_spin.setEnabled(False)
        self.blur_label = QLabel(self.tr("Blur radius (in pixels):"))
        self.blur_label.setEnabled(False)
        param_layout.addRow(self.blur_label, self.blur_spin)

        # Methods
        method_widget = QWidget()
        method_layout = QHBoxLayout(method_widget)
        self.check_vo = QCheckBox("VO", self)
        self.check_vo.setChecked(True)
        self.check_vo.setEnabled(False)
        self.check_vop = QCheckBox("VOP", self)
        self.check_von = QCheckBox("VON", self)
        method_layout.addWidget(self.check_vo)
        method_layout.addWidget(self.check_vop)
        method_layout.addWidget(self.check_von)
        method_layout.addStretch()
        param_layout.addRow(method_widget)

        layout.addWidget(param_box)

        # Output options
        output_box = QGroupBox(self.tr("Output parameters"))
        output_layout = QFormLayout(output_box)
        self.check_8bits = QCheckBox(self.tr("8-bits output"), self)
        self.check_8bits.setChecked(True)
        output_layout.addRow(self.check_8bits)
        layout.addWidget(output_box)

        # GPU
        self.check_gpu = QCheckBox(self.tr("Use GPU"), self)
        self.check_gpu.setEnabled(GPU_AVAILABLE)
        self.check_gpu.setChecked(GPU_AVAILABLE)
        layout.addWidget(self.check_gpu)

        # Progress
        progress_box = QGroupBox(self.tr("Progress"))
        progress_layout = QVBoxLayout(progress_box)
        self.file_label = QLabel(self.tr("Ready"))
        progress_layout.addWidget(self.file_label)
        self.file_progress = QProgressBar(self)
        self.file_progress.setMaximum(len(self.file_list))
        progress_layout.addWidget(self.file_progress)
        self.pixel_progress = QProgressBar(self)
        self.pixel_progress.setMaximum(100)
        progress_layout.addWidget(self.pixel_progress)
        layout.addWidget(progress_box)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton(self.tr("Start"), self)
        self.btn_start.clicked.connect(self.start_batch)
        self.btn_close = QPushButton(self.tr("Close"), self)
        self.btn_close.clicked.connect(self.close)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        self._update_radius_m_from_res()

    def _get_first_x_res(self):
        if not self.file_list:
            return 1.0
        try:
            im = _open_raster(self.file_list[0])
            gt = im.GetGeoTransform()
            if gt is None:
                gt = (0, 1, 0, 0, 0, -1)
            x_res = float(abs(gt[1]))
            if x_res == 0:
                x_res = 1.0
            return x_res
        except Exception:
            return 1.0

    def _radius_unit_changed(self, index):
        if index == 0:
            self.radius_spin.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.radius_m.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        else:
            self.radius_m.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.radius_spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.radius_stack.setCurrentIndex(index)
        self.radius_stack.updateGeometry()
        self._sync_radius_m_from_px()

    def _radius_px_changed(self, value):
        self._sync_radius_m_from_px()

    def _radius_m_changed(self, value):
        x_res = getattr(self, 'x_res', None)
        if not x_res or x_res <= 0:
            return
        px = max(2, int(round(value / x_res)))
        max_px = self.radius_spin.maximum()
        px = min(px, max_px)
        self.radius_spin.blockSignals(True)
        self.radius_spin.setValue(px)
        self.radius_spin.blockSignals(False)
        self._sync_radius_m_from_px()

    def _sync_radius_m_from_px(self):
        x_res = getattr(self, 'x_res', None)
        if not x_res or x_res <= 0:
            return
        m_value = self.radius_spin.value() * x_res
        self.radius_m.blockSignals(True)
        self.radius_m.setValue(m_value)
        self.radius_m.blockSignals(False)

    def _update_radius_m_from_res(self):
        x_res = getattr(self, 'x_res', None)
        if not x_res or x_res <= 0:
            return
        px_max = self.radius_spin.maximum()
        self.radius_m.setRange(2 * x_res, px_max * x_res)
        decs = max(3, 1 + int(math.ceil(-math.log10(x_res))))
        self.radius_m.setDecimals(decs)
        self.radius_m.setSingleStep(x_res)
        self._sync_radius_m_from_px()

    def toggle_blur(self):
        self.blur_spin.setEnabled(self.check_blur.isChecked())
        self.blur_label.setEnabled(self.check_blur.isChecked())

    def start_batch(self):
        self.btn_start.setEnabled(False)
        self.batch_calc = BatchCalculation(
            self.file_list,
            self.radius_spin.value(),
            self.exageration_spin.value(),
            self.check_blur.isChecked(),
            self.blur_spin.value(),
            self.check_vop.isChecked(),
            self.check_von.isChecked(),
            self.check_gpu.isChecked(),
            self.check_8bits.isChecked()
        )
        self.batch_calc.fileChanged.connect(self.on_file_changed)
        self.batch_calc.pixelProgress.connect(self.on_pixel_progress)
        self.batch_calc.finished.connect(self.on_finished)
        self.batch_calc.start()

    def on_file_changed(self, index, filename):
        self.file_label.setText(self.tr("Processing: {}").format(filename))
        self.file_progress.setValue(index)
        self.pixel_progress.setValue(0)

    def on_pixel_progress(self, value):
        self.pixel_progress.setValue(value)

    def on_finished(self):
        self.file_progress.setValue(len(self.file_list))
        self.pixel_progress.setValue(self.pixel_progress.maximum())
        self.file_label.setText(self.tr("Batch complete! {} files processed.").format(len(self.file_list)))
        self.btn_start.setEnabled(True)
        QMessageBox.information(self, self.tr("Batch Processing"), 
                               self.tr("All {} DEMs have been processed.").format(len(self.file_list)))


class BatchCalculation(QThread):

    fileChanged = pyqtSignal(int, str)
    pixelProgress = pyqtSignal(int)

    def __init__(self, file_list, radius, exageration, do_blur, blur_radius,
                 checked_vop, checked_von, use_cuda, save_8bits):
        super(BatchCalculation, self).__init__()
        self.file_list = file_list
        self.radius = radius
        self.exageration = exageration
        self.do_blur = do_blur
        self.blur_radius = blur_radius
        self.checked_vop = checked_vop
        self.checked_von = checked_von
        self.use_cuda = use_cuda
        self.save_8bits = save_8bits

    def run(self):
        use_gpu = GPU_AVAILABLE and self.use_cuda

        # Build grid
        x = np.arange(-self.radius, self.radius + 1, dtype=int)
        y = np.arange(-self.radius, self.radius + 1, dtype=int)
        grid = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)
        distance = np.sqrt(grid[:, 0]**2 + grid[:, 1]**2)
        d = distance[distance < self.radius]
        grid = grid[distance < self.radius]

        for file_idx, image_path in enumerate(self.file_list):
            image_folder, image_name = os.path.split(image_path)
            self.fileChanged.emit(file_idx, image_name)

            # Open DEM
            try:
                im_originale = _open_raster(image_path)
            except Exception as exc:
                print(self.tr("Skipping {}: {}").format(image_name, exc))
                continue

            gt = im_originale.GetGeoTransform()
            if gt is None or gt[1] == 0 or gt[5] == 0:
                gt = (0, 1, 0, 0, 0, -1)
            x_res = gt[1]

            # Vertical correction
            h_vert_cor = np.sqrt(self.radius**2 - d**2) * x_res / self.exageration

            # Read and prepare image
            im_numpy = im_originale.GetRasterBand(1).ReadAsArray()
            wherenan, image_final_npy = tranformImage(im_numpy)
            image_final_npy = np.nan_to_num(image_final_npy)

            # Optional blur
            if self.do_blur:
                image_final_npy = blurring(image_final_npy, self.blur_radius)

            # Calculation
            sum_h = float(h_vert_cor.sum())
            last_progress = -1

            def progress_callback(i):
                nonlocal last_progress
                progress = int((i + 1) * 100 / d.size)
                if progress != last_progress:
                    self.pixelProgress.emit(progress)
                    last_progress = progress

            if use_gpu and TAICHI_AVAILABLE:
                acc_vo, acc_vop = _compute_taichi(
                    image_final_npy, h_vert_cor, grid, progress_callback=progress_callback
                )
            else:
                acc_vo, acc_vop = _compute_numpy(
                    image_final_npy, h_vert_cor, grid, progress_callback=progress_callback
                )

            out_vo = acc_vo / (2.0 * sum_h)
            out_vop = acc_vop / sum_h
            out_von = -(acc_vo - acc_vop - sum_h) / sum_h

            # Save results
            basename = os.path.splitext(image_name)[0]
            exagere = '' if self.exageration == 1 else f'_Z-exag={self.exageration}'
            blur_str = f'_smooth_r={self.blur_radius}' if self.do_blur else ''

            cols = im_originale.RasterXSize
            rows = im_originale.RasterYSize
            bands = 1
            gt = gt  # already normalized above
            proj = im_originale.GetProjection() or _local_wkt()

            def save_single(data, label):
                data[wherenan] = np.nan
                outname = os.path.join(image_folder, f'{basename}_{label}_r={self.radius}{blur_str}{exagere}.tif')
                driver = gdal.GetDriverByName('GTIFF')
                driver.Register()
                output = driver.Create(outname, cols, rows, bands, gdal.GDT_Float32)
                output.SetGeoTransform(gt)
                output.SetProjection(proj)
                outBand = output.GetRasterBand(1)
                outBand.WriteArray(data, 0, 0)
                output.FlushCache()
                output = None
                outBand = None
                if self.save_8bits:
                    img8 = array_image(data, 2.5, 97.5).astype(np.uint8)
                    outname8 = os.path.join(image_folder, f'{basename}_{label}_8bits_r={self.radius}{blur_str}{exagere}.jpg')
                    Image.fromarray(img8).save(outname8)

            # VO
            save_single(out_vo, 'VO')

            # VOP
            if self.checked_vop:
                save_single(out_vop, 'VOP')

            # VON
            if self.checked_von:
                save_single(out_von, 'VON')

            # RGB composite
            if self.checked_vop and self.checked_von:
                r_channel = array_image(out_vo, 2.5, 97.5)
                g_channel = array_image(out_vop, 2.5, 97.5)
                b_channel = array_image(out_von, 2.5, 97.5)
                rgb = np.dstack((r_channel, g_channel, b_channel)).astype(np.uint8)
                if self.save_8bits:
                    rgb_name = os.path.join(image_folder, f'{basename}_RGB_combine_8bits_r={self.radius}{blur_str}{exagere}.jpg')
                    Image.fromarray(rgb, mode='RGB').save(rgb_name)

            im_originale = None

        print("Batch processing complete.")


def blurring(in_array, size):

    # Blur using fft
    
    padded_array = np.pad(in_array, size, 'symmetric')
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
    g = (g / g.sum()).astype(in_array.dtype)
    
    return fftconvolve(padded_array, g, mode='valid')
      


def tranformImage(image):
    
    # transform DEM in 32 bits and put Nan in place of +/- 32767
    
    image = np.float32(image)
    image[image == -32767] = 'nan'
    image[image == 32767] = 'nan'
    wherenan = np.isnan(image)
    image_final = image
    
    return wherenan, image_final # return the palce an array witht the position of Nan and the transformed DEM
        

def array_image(image_input, p1, p2):
    
    # contrast improvement within the p1 - p2 percentiles (truncate)
    
    percentile = np.nanpercentile(image_input, [p1, p2])
    
    image_input = np.nan_to_num(image_input)
    image_input[image_input < percentile[0]] = percentile[0]
    image_input[image_input > percentile[1]] = percentile[1]
    
    image_input = image_input - np.min(image_input)
    max_value = np.max(image_input)
    if max_value > 0:
        image_input = image_input / max_value
    image_input = 255 * image_input
    out = image_input.astype(np.uint8)
    
    return out # return the contrasted image
        

def _round_to_first_sig(x):
    if x <= 0:
        return 0.0
    d = math.floor(math.log10(x))
    return round(x, -d)


def main():
    # Initialize Taichi on the main thread before any QThread uses it.
    _init_taichi()
    _taichi_precompile()
    if '--smoke-test' in sys.argv:
        if not TAICHI_AVAILABLE:
            raise RuntimeError('Taichi is unavailable')
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName("vSky2")
    app.setApplicationDisplayName("vSky2")
    app.setOrganizationName("Université de Bourgogne")
    QLocale.setDefault(QLocale.c())
    locale = QLocale.system().name()
    print(locale)
    qtTranslator = QTranslator()
    if qtTranslator.load("qt_" + locale, ":/"):
        app.installTranslator(qtTranslator)
    appTranslator = QTranslator()
    if appTranslator.load("vSky_" + locale, "./"):
        app.installTranslator(appTranslator)
    win = MainWindow()
    gdal.SetErrorHandler(_gdal_error_handler)
    win.show()
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main()) 
