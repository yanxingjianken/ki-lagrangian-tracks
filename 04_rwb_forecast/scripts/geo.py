# Copy of roger/03_era5/scripts/geo.py.
"""North-polar stereographic projection (tangent at the pole) and great-circle distance on a 6371-km sphere."""
import numpy as np


def stereo_to_lonlat(x_km):
    """Inverse north-polar stereographic projection (tangent at the pole) -> lon, lat in degrees."""
    r = np.hypot(x_km[..., 0], x_km[..., 1])
    lat = 90.0 - np.rad2deg(2 * np.arctan(r / (2 * 6371.0)))
    lon = np.rad2deg(np.arctan2(x_km[..., 0], -x_km[..., 1])) % 360.0
    return lon, lat


def lonlat_to_stereo(lon, lat):
    """North-polar stereographic projection (tangent at the pole) in km."""
    r = 2 * 6371.0 * np.tan(np.deg2rad(90.0 - lat) / 2)
    lam = np.deg2rad(lon)
    return np.stack([r * np.sin(lam), -r * np.cos(lam)], axis=-1)


def great_circle_km(lon1, lat1, lon2, lat2):
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return 6371.0 * np.arccos(np.clip(c, -1.0, 1.0))
