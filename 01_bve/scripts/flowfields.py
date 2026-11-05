"""Streamfunction of a wind field on a regular latitude-longitude grid (spherical harmonics, torch-harmonics)."""
import numpy as np
import torch
import torch_harmonics as th

A_M = 6.37122e6


def streamfunction(u, v):
    """psi (m2/s) on the regular 1-degree grid (lat 90..-90): the first vector-harmonic potential of (u, v) times a."""
    nlat, nlon = u.shape[-2:]
    vsht = th.RealVectorSHT(nlat, nlon, grid="equiangular")
    isht = th.InverseRealSHT(nlat, nlon, lmax=vsht.lmax, mmax=vsht.mmax, grid="equiangular")
    coef = vsht(torch.tensor(np.stack([u, v], -3), dtype=torch.float64))[..., 0, :, :]
    return isht(coef * A_M).numpy()
