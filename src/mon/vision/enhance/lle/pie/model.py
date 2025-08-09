#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PIE model for low-light image enhancement.

References:
    - Paper: "A Probabilistic Method for Image Enhancement With Simultaneous
      Illumination and Reflectance Estimation," IEEE TIP 2015.
    - Code: https://github.com/DavidQiuChao/PIE
"""

__all__ = [
    "PIE",
]

import box
import cv2
import numpy as np

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


def calc_eta(R, nR, eta):
    f  = np.linalg.norm(nR - R, ord=2)
    f /= (np.linalg.norm(R, ord=2) + 1e-10)
    if f <= eta:
        return 1
    return 0


def image_rad(im):
    gv = cv2.Sobel(im, cv2.CV_32F, 1, 0)
    gh = cv2.Sobel(im, cv2.CV_32F, 0, 1)
    return gv, gh


def psf2otf(psf, outSize):
    '''
    code is from https://blog.csdn.net/
    weixin_43890288/article/details/
    105676416
    '''
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf     = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), "constant")
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf   = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps  = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps  = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf


def shrink(x, lam):
    eps   = 1e-10
    abs_x = abs(x)
    f     = x * 1.0 / (abs_x + eps)
    f    *= np.maximum(abs_x - lam, 0)
    return f


def upDateP1(R, bv, bh, lam):
    dvR, dhR = image_rad(R)
    dv = shrink(dvR + bv, 1.0 / (2 * lam))
    dh = shrink(dhR + bh, 1.0 / (2 * lam))
    return dv, dh


def upDataP2(S, I, difv, difh, fdH, fdV, fdHcj, fdVcj, beta, lam):
    eps   = 1e-10
    ahp   = beta * lam
    dfdvR = np.fft.fft2(difv)
    dfdhR = np.fft.fft2(difh)
    Fi    = fdVcj * dfdvR + fdHcj * dfdhR
    f1    = np.fft.fft2(S / (I + eps))
    f1   += ahp * Fi
    f2    = (np.abs(fdH) ** 2 + np.abs(fdV) ** 2) * ahp + 1
    R     = np.fft.ifft2(f1 / f2)
    R     = np.abs(R).astype(np.float32)
    R     = np.maximum(0, np.minimum(1, R))
    dvR, dhR = image_rad(R)
    bh    = dhR - difh
    bv    = dvR - difv
    return R, bv, bh


def upDataP3(S, R, I0, gama, alpha, fdH, fdV, fdHcj, fdVcj):
    eps = 1e-10
    f1  = gama * I0 + S / (R + eps)
    f1  = np.fft.fft2(f1)
    f2  = alpha * (np.abs(fdH) ** 2 + np.abs(fdV) ** 2) + gama + 1
    I   = np.fft.ifft2(f1 / f2)
    I   = np.abs(I)
    I   = np.maximum(0, np.minimum(255.0, I))
    I   = np.maximum(I, S).astype(np.float32)
    return I


def optimizAlgo(S, alpha, beta, lam, gama, eta1, eta2):
    # initialization
    I     = cv2.GaussianBlur(S, (5, 5), 0)
    I0    = np.mean(S)
    R     = np.zeros(I.shape,dtype=np.float32)
    bv    = 0
    bh    = 0
    H, W  = I.shape
    # FFT and conjugate for derivate operators
    sbV   = np.array([[1, -1]])
    sbH   = np.array([[1], [-1]])
    fdH   = psf2otf(sbH, (H + 1, W))
    fdV   = psf2otf(sbV, (H, W +  1))
    fdV   = fdV[:, 1:]
    fdH   = fdH[1:, :]
    fdHcj = np.conjugate(fdH)
    fdVcj = np.conjugate(fdV)

    # begin main iteration
    idx = 0
    while 1:
        dv, dh = upDateP1(R, bv, bh, lam)
        difv = dv - bv
        difh = dh - bh
        nR, bv, bh = upDataP2(S, I, difv, difh, fdH, fdV, fdHcj, fdVcj, beta, lam)
        tR = calc_eta(R, nR, eta1)
        R  = nR
        nI = upDataP3(S, R, I0, gama, alpha, fdH, fdV, fdHcj, fdVcj)
        tI = calc_eta(I, nI, eta2)
        I  = nI
        idx += 1
        # teminate criterion
        if tR and tI:
            break
        if idx > 1:
            break
    # gamma correction
    I = 255.0 * np.power(I / 255.0, 1 / 2.2)
    return R * I


'''
def PIE(src):
    alpha = 1000
    beta  = 0.01
    lam   = 10
    gama  = 0.1
    eta1  = 0.1
    eta2  = 0.1
    im    = src[:, :, ::-1]
    hsv   = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    v     = hsv[:, :, 2].astype(np.float32)
    v     = optimizAlgo(v, alpha, beta, lam, gama, eta1, eta2)
    hsv[:, :, 2] = v
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    im = np.minimum(255, np.maximum(0, im))
    im = im[:, :, ::-1]
    return im
'''


@MODELS.register(name="pie", arch="pie")
class PIE(nn.Module, nn.ModelMixin):
    """PIE model for low-light image enhancement.
    
    References:
        - Paper: "A Probabilistic Method for Image Enhancement With Simultaneous
          Illumination and Reflectance Estimation," IEEE TIP 2015.
        - Code: https://github.com/DavidQiuChao/PIE
    """
    
    arch     : str          = "pie"
    name     : str          = "pie"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.TRADITIONAL]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()

    def __init__(self):
        super().__init__()
        self.alpha = 1000
        self.beta  = 0.01
        self.lam   = 10
        self.gama  = 0.1
        self.eta1  = 0.1
        self.eta2  = 0.1
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        v   = hsv[:, :, 2].astype(np.float32)
        v   = optimizAlgo(v, self.alpha, self.beta, self.lam, self.gama, self.eta1, self.eta2)
        hsv[:, :, 2] = v
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb = np.minimum(255, np.maximum(0, rgb))
        return rgb
