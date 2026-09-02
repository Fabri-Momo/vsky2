import os
import numpy as np
from osgeo import gdal, osr
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ModuleNotFoundError:
    CUDA_AVAILABLE = False


def blurring(in_array, size):
    # Blur using fft
    padded_array = np.pad(in_array, size, 'symmetric')
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
    g = (g / g.sum()).astype(in_array.dtype)

    windows = np.lib.stride_tricks.sliding_window_view(padded_array, g.shape)
    return np.einsum('ijkl,kl->ij', windows, g, optimize=True)


def _vo(feedback, array, grid, distance, vertical_correction):
    progress_max = 100 / (distance.size - 1)
    feedback.pushInfo('Computing vo')
    large_vo = np.zeros((array.shape[0], array.shape[1], 2))
    for i in range(0, distance.size):
        M = (np.roll(array, [grid[i, 0], grid[i, 1]], axis = [0, 1]) - array)
        vop = vertical_correction[i] - M
        vop[vop > 2 * vertical_correction[i]] = 2 * vertical_correction[i]
        vop[vop < 0] = 0
        large_vo[:,:,1] = vop
        large_vo[:,:,0]  = np.sum(large_vo, axis=2)  #cumulative sum
        feedback.setProgress(i * progress_max)
    return large_vo


def _vop(feedback, array, grid, distance, vertical_correction):
    progress_max = 100 / (distance.size - 1)
    feedback.pushInfo('Computing vop')
    large_vo = np.zeros((array.shape[0], array.shape[1], 2))
    large_vop = np.zeros((array.shape[0], array.shape[1], 2))
    for i in range(0, distance.size):
        M = (np.roll(array, [grid[i, 0], grid[i, 1]], axis = [0, 1]) - array)
        vop = vertical_correction[i] - M
        vop[vop > 2 * vertical_correction[i]] = 2 * vertical_correction[i]
        vop[vop < 0] = 0
        large_vo[:,:,1] = vop
        large_vo[:,:,0]  = np.sum(large_vo, axis=2)  #cumulative sum
        svf_vop = np.copy(vop)
        svf_vop[svf_vop > vertical_correction[i]] = vertical_correction[i]
        large_vop[:,:,1] = svf_vop
        large_vop[:,:,0] = np.sum(large_vop, axis=2)  #cumulative sum
        feedback.setProgress(i * progress_max)
    return large_vop


def _von(feedback, array, grid, distance, vertical_correction):
    progress_max = 100 / (distance.size - 1)
    feedback.pushInfo('Computing von')
    large_vo = np.zeros((array.shape[0], array.shape[1], 2))
    large_vop = np.zeros((array.shape[0], array.shape[1], 2))
    large_von = np.zeros((array.shape[0], array.shape[1], 2))
    for i in range(0, distance.size):
        M = (np.roll(array, [grid[i, 0], grid[i, 1]], axis = [0, 1]) - array)
        vop = vertical_correction[i] - M
        vop[vop > 2 * vertical_correction[i]] = 2 * vertical_correction[i]
        vop[vop < 0] = 0
        large_vo[:,:,1] = vop
        large_vo[:,:,0]  = np.sum(large_vo, axis=2)  #cumulative sum
        svf_vop = np.copy(vop)
        svf_vop[svf_vop > vertical_correction[i]] = vertical_correction[i]
        large_vop[:,:,1] = svf_vop
        large_vop[:,:,0] = np.sum(large_vop, axis=2)  #cumulative sum
        large_von[:,:,0] = - (large_vo[:,:,0] - large_vop[:,:,0] - vertical_correction.sum()) #von obtained by subtraction
        feedback.setProgress(i * progress_max)
    return large_von


def _vo_on_cuda(feedback, array, grid, distance, vertical_correction):
    progress_max = 100 / (distance.size - 1)
    feedback.pushInfo('Computing vo')
    large_vo = cp.zeros((array.shape[0], array.shape[1], 2))
    for i in range(0, distance.size):
        M = (cp.roll(array, [grid[i, 0], grid[i, 1]], axis = [0, 1]) - array)
        cp.cuda.Stream.null.synchronize()
        vop = vertical_correction[i] - M
        vop[vop > 2 * vertical_correction[i]] = 2 * vertical_correction[i]
        vop[vop < 0] = 0
        large_vo[:,:,1] = vop
        large_vo[:,:,0]  = cp.sum(large_vo, axis=2)  #cumulative sum
        cp.cuda.Stream.null.synchronize()
        feedback.setProgress(i * progress_max)
    return large_vo


def _vop_on_cuda(feedback, array, grid, distance, vertical_correction):
    progress_max = 100 / (distance.size - 1)
    feedback.pushInfo('Computing vop')
    large_vo = cp.zeros((array.shape[0], array.shape[1], 2))
    large_vop = cp.zeros((array.shape[0], array.shape[1], 2))
    for i in range(0, distance.size):
        M = (cp.roll(array, [grid[i, 0], grid[i, 1]], axis = [0, 1]) - array)
        cp.cuda.Stream.null.synchronize()
        vop = vertical_correction[i] - M
        vop[vop > 2 * vertical_correction[i]] = 2 * vertical_correction[i]
        vop[vop < 0] = 0
        large_vo[:,:,1] = vop
        large_vo[:,:,0]  = cp.sum(large_vo, axis=2)  #cumulative sum
        svf_vop = cp.copy(vop)
        svf_vop[svf_vop > vertical_correction[i]] = vertical_correction[i]
        large_vop[:,:,1] = svf_vop
        large_vop[:,:,0] = cp.sum(large_vop, axis=2)  #cumulative sum
        cp.cuda.Stream.null.synchronize()
        feedback.setProgress(i * progress_max)
    return large_vop


def _von_on_cuda(feedback, array, grid, distance, vertical_correction):
    progress_max = 100 / (distance.size - 1)
    feedback.pushInfo('Computing von')
    large_vo = cp.zeros((array.shape[0], array.shape[1], 2))
    large_vop = cp.zeros((array.shape[0], array.shape[1], 2))
    large_von = cp.zeros((array.shape[0], array.shape[1], 2))
    for i in range(0, distance.size):
        M = (cp.roll(array, [grid[i, 0], grid[i, 1]], axis = [0, 1]) - array)
        cp.cuda.Stream.null.synchronize()
        vop = vertical_correction[i] - M
        vop[vop > 2 * vertical_correction[i]] = 2 * vertical_correction[i]
        vop[vop < 0] = 0
        large_vo[:,:,1] = vop
        large_vo[:,:,0]  = cp.sum(large_vo, axis=2)  #cumulative sum
        svf_vop = cp.copy(vop)
        svf_vop[svf_vop > vertical_correction[i]] = vertical_correction[i]
        large_vop[:,:,1] = svf_vop
        large_vop[:,:,0] = cp.sum(large_vop, axis=2)  #cumulative sum
        large_von[:,:,0] = - (large_vo[:,:,0] - large_vop[:,:,0] - vertical_correction.sum()) #von obtained by subtraction
        cp.cuda.Stream.null.synchronize()
        feedback.setProgress(i * progress_max)
    return large_von


def vsky_compute(feedback, mode, input_file, output_file, smoothing=0, z_factor=1, radius=5, use_cuda=False):
    ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError('Unable to open the input raster')
    crs = ds.GetProjectionRef()
    proj = ds.GetProjection()
    srs = osr.SpatialReference(wkt = proj)
    feedback.pushInfo('{}'.format(srs))
    if not srs.IsProjected():
        raise ValueError('The input raster must use a projected coordinate reference system')
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    out = band.ReadAsArray()
    wherenan = np.isnan(out)
    out = np.nan_to_num(out)
    dtype = out.dtype
    if dtype != out.dtype:
        out = out.astype(dtype, copy=False)

    x = np.arange(-radius, radius + 1, dtype = int)
    y = np.arange(-radius, radius + 1, dtype = int)
    grid = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)
    distance = np.sqrt(grid[:,0]**2 + grid[:,1]**2)
    d = distance[distance < radius]
    grid = grid[distance < radius]
    h_vert_cor = np.sqrt(radius**2 - d**2) * gt[1] / z_factor
    if use_cuda:
        if smoothing > 0:
            feedback.pushInfo('Applying low-pass filter')
            out = blurring(out, smoothing)
        out = cp.asarray(out)
        if mode == 'vo':
            out = _vo_on_cuda(feedback, out, grid, d, h_vert_cor)
            out = out[:,:,0]  / (2 * h_vert_cor.sum())
        elif mode == 'vop':
            out = _vop_on_cuda(feedback, out, grid, d, h_vert_cor)
            out = out[:,:,0]  / h_vert_cor.sum()
        elif mode == 'von':
            out = _von_on_cuda(feedback, out, grid, d, h_vert_cor)
            out = out[:,:,0]  / h_vert_cor.sum()
        out = cp.asnumpy(out)
    else:
        if smoothing > 0:
            feedback.pushInfo('Applying low-pass filter')
            out = blurring(out, smoothing)
        if mode == 'vo':
            out = _vo(feedback, out, grid, d, h_vert_cor)
            out = out[:,:,0]  / (2 * h_vert_cor.sum())
        elif mode == 'vop':
            out = _vop(feedback, out, grid, d, h_vert_cor)
            out = out[:,:,0]  / h_vert_cor.sum()
        elif mode == 'von':
            out = _von(feedback, out, grid, d, h_vert_cor)
            out = out[:,:,0]  / h_vert_cor.sum()
    out[wherenan == True] = 'nan'
    h,w = out.shape
    drv = gdal.GetDriverByName("GTiff")
    ds_out  = drv.Create(output_file, w, h, 1, gdal.GDT_Float32,
            """COMPRESS=DEFLATE
                ZLEVEL=4
                BIGTIFF=IF_SAFER
                PREDICTOR=3
                NUM_THREADS=ALL_CPUS""".split())
    ds_out.SetProjection(crs)
    ds_out.SetGeoTransform(gt)
    band = ds_out.GetRasterBand(1)
    band.WriteArray(out)
    #band.SetNoDataValue("nan")
    ds_out.FlushCache()
