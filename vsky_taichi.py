import taichi as ti


@ti.kernel
def accumulate_chunk(
    image: ti.types.ndarray(ndim=2),
    grid: ti.types.ndarray(ndim=2),
    h: ti.types.ndarray(ndim=1),
    start: ti.i32,
    end: ti.i32,
    acc_vo: ti.types.ndarray(ndim=2),
    acc_vop: ti.types.ndarray(ndim=2),
):
    H = image.shape[0]
    W = image.shape[1]
    for x, y in acc_vo:
        vo = 0.0
        vop = 0.0
        for i in range(start, end):
            dx = grid[i, 0]
            dy = grid[i, 1]
            nx = (x + dx) % H
            ny = (y + dy) % W
            diff = h[i] - (image[nx, ny] - image[x, y])
            if diff < 0.0:
                diff = 0.0
            elif diff > 2.0 * h[i]:
                diff = 2.0 * h[i]
            vo += diff
            svf = diff
            if svf > h[i]:
                svf = h[i]
            vop += svf
        acc_vo[x, y] += vo
        acc_vop[x, y] += vop
