from __future__ import absolute_import, division, print_function

import numpy as np
import png
import matplotlib.colors as cl

TAG_CHAR = np.array([202021.25], np.float32)
UNKNOWN_FLOW_THRESH = 1e7


def write_flow(filename, uv, v=None):
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def write_flow_png(filename, uv, v=None, mask=None):

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        if uv.min() < -512. or uv.max() > 512.:
            print("Clipping uv to [-512, 512]!")
            uv = np.clip(uv, -512., 512.)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)

    height_img, width_img = u.shape
    if mask is None:
        valid_mask = np.ones([height_img, width_img])
    else:
        valid_mask = mask

    flow_u = np.clip((u * 64. + 2. ** 15.), 0.0, 65535.0).astype(np.float64)
    flow_v = np.clip((v * 64. + 2. ** 15.), 0.0, 65535.0).astype(np.float64)

    output = np.stack((flow_u, flow_v, valid_mask), axis=-1)

    with open(filename, 'wb') as f:
        writer = png.Writer(width=width_img, height=height_img, bitdepth=16)
        to_write = np.reshape(output, (-1, width_img*3)).round()
        writer.write(f, to_write)


def write_flow_png2(name, flow):
    H, W, _ = flow.shape
    out = np.ones((H, W, 3), dtype=np.float64)
    out[:,:,1] = np.minimum(np.maximum(flow[:, :, 1] * 64. + 2**15, 0.), 2**16-1).astype(np.float64)#.astype(np.uint64)
    out[:,:,0] = np.minimum(np.maximum(flow[:, :, 0] * 64. + 2**15, 0.), 2**16-1).astype(np.float64)#.astype(np.uint64)

    recon_uv = (out[:, :, 0:2] - 2. ** 15.) / 64.0
    diffbw = (recon_uv - flow).sum()
    print('Diff_before_write {}'.format(diffbw))

    with open(name, 'wb') as f:
        writer = png.Writer(width=W, height=H, bitdepth=16)
        im2list = out.reshape(-1, out.shape[1]*out.shape[2]).round().tolist()
        writer.write(f, im2list)


def write_flow_png3(name, flow):
    H, W, _ = flow.shape
    out = np.ones((H, W, 3), dtype=np.uint64)
    out[:,:,1] = np.minimum(np.maximum(flow[:,:,1]*64.+2**15, 0), 2**16-1).astype(np.uint64)
    out[:,:,0] = np.minimum(np.maximum(flow[:,:,0]*64.+2**15, 0), 2**16-1).astype(np.uint64)
    with open(name, 'wb') as f:
        writer = png.Writer(width=W, height=H, bitdepth=16)
        im2list = out.reshape(-1, out.shape[1]*out.shape[2]).tolist()
        writer.write(f, im2list)


def write_flow_png_orig(filename, uv, v=None, mask=None):
    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)

    height_img, width_img = u.shape
    if mask is None:
        valid_mask = np.ones([height_img, width_img])
    else:
        valid_mask = mask

    flow_u = np.clip((u * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)
    flow_v = np.clip((v * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)

    output = np.stack((flow_u, flow_v, valid_mask), axis=-1)

    with open(filename, 'wb') as f:
        writer = png.Writer(width=width_img, height=height_img, bitdepth=16)
        writer.write(f, np.reshape(output, (-1, width_img * 3)))


def flow_to_png(flow_map, max_value=None):
    _, h, w = flow_map.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[:, :, 0] += normalized_flow_map[0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:, :, 2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)



def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def flow_to_png_middlebury(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """

    flow = flow.transpose([1, 2, 0])
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)