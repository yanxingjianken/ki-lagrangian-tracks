# Copy of roger/01_bve/scripts/flowfields.py, plus the vorticity and the hemispheric mirror used here.
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


def vorticity(u, v):
    """Relative vorticity (1/s) on the regular grid (lat 90..-90) from the vector-harmonic potential of (u, v)."""
    nlat, nlon = u.shape[-2:]
    vsht = th.RealVectorSHT(nlat, nlon, grid="equiangular")
    isht = th.InverseRealSHT(nlat, nlon, lmax=vsht.lmax, mmax=vsht.mmax, grid="equiangular")
    coef = vsht(torch.tensor(np.stack([u, v], -3), dtype=torch.float64))[..., 0, :, :]
    l = torch.arange(vsht.lmax, dtype=torch.float64)[:, None]
    return isht(-l * (l + 1) * coef / A_M).numpy()


def mirror_global(u, v, lat):
    """Global fields (lat 90..-90) from northern-hemisphere u, v on lat 0..90 (u symmetric, v antisymmetric)."""
    north = np.argsort(lat)[::-1]
    un, vn = u[..., north, :], v[..., north, :]
    return np.concatenate([un, un[..., -2::-1, :]], -2), np.concatenate([vn, -vn[..., -2::-1, :]], -2)
