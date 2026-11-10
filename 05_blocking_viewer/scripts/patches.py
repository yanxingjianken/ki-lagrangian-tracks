# Copy of roger/03_era5/scripts/patches.py.
"""Parcel patches like the ocean data: compact disks of parcels released together at random, well-separated centres."""
import numpy as np

A_KM = 6371.0


def destination(lat, lon, bearing, dist_km):
    """Point reached from (lat, lon) [deg] along a great circle with the given bearing [rad] and distance."""
    p1, l1, d = np.deg2rad(lat), np.deg2rad(lon), dist_km / A_KM
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(bearing))
    l2 = l1 + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.rad2deg(p2), np.rad2deg(l2) % 360.0


def seed_patches(n_patches=8, n_per=1000, radius_km=300.0, lat_range=(35.0, 60.0), min_sep_km=2000.0, seed=0):
    """Centres uniform in area within lat_range, at least min_sep_km apart; parcels uniform in each disk.

    Returns lat, lon (n_patches * n_per,), patch index per parcel, and the centres (n_patches, 2) as (lat, lon).
    """
    rng = np.random.default_rng(seed)
    s0, s1 = np.sin(np.deg2rad(lat_range[0])), np.sin(np.deg2rad(lat_range[1]))
    centres = []
    while len(centres) < n_patches:
        c = (np.rad2deg(np.arcsin(rng.uniform(s0, s1))), rng.uniform(0.0, 360.0))
        far = all(_gc(c, q) >= min_sep_km for q in centres)
        if far:
            centres.append(c)
    lat, lon, pid = [], [], []
    for k, (clat, clon) in enumerate(centres):
        r = radius_km * np.sqrt(rng.uniform(0.0, 1.0, n_per))
        la, lo = destination(clat, clon, rng.uniform(0.0, 2 * np.pi, n_per), r)
        lat.append(la)
        lon.append(lo)
        pid.append(np.full(n_per, k))
    return np.concatenate(lat), np.concatenate(lon), np.concatenate(pid), np.asarray(centres)


def _gc(a, b):
    p1, p2, dl = np.deg2rad(a[0]), np.deg2rad(b[0]), np.deg2rad(b[1] - a[1])
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return A_KM * np.arccos(np.clip(c, -1.0, 1.0))
