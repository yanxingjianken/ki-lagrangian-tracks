"""Checks of the release schedule, the hemisphere mirror and the minibatch sampler of 03_era5.

Run:  micromamba run -n kinematic_interpolants python -m pytest -q tests/test_pipeline.py
Scratch files go to a new directory under $ROGER_TEST_DIR (default: data/test_scratch of this folder).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ctx_data import load_patches, sample_batch, sin_lat  # noqa: E402
from geo import lonlat_to_stereo, stereo_to_lonlat  # noqa: E402

BASE = os.environ.get("ROGER_TEST_DIR", str(ROOT / "data/test_scratch"))


def load(name):
    spec = importlib.util.spec_from_file_location(name[:-3].lstrip("0123456789_"), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_release_schedule():
    cfg = json.loads((ROOT / "configs/releases.json").read_text())
    rel = load("00_releases.py").releases()
    assert [r["name"] for r in rel] == [r["name"] for r in cfg["releases"]]
    for split, n, extra in (("train", 220, 0), ("test", 73, 8)):
        r = [x for x in cfg["releases"] if x["split"] == split]
        t = np.array([np.datetime64(x["time"], "h") for x in r])
        assert len(r) == n
        assert np.all((t - t[0]).astype(int) == 120 * np.arange(n) + 6 * (np.arange(n) % 4))
        assert all(sum(b[1] for b in x["bands"]) == 16 + extra for x in r)
        assert all((np.datetime64(x["end"]) - np.datetime64(x["time"])).astype(int) == 156 for x in r)
    assert len(cfg["sigma"]) == 20 and {0.967, 0.887, 0.674, 0.477, 0.241} <= set(cfg["sigma"])
    sp = cfg["split_by_patch"]
    assert not set(sp["train"]) & (set(sp["validation"]) | set(sp["test"]) | set(sp["unseen_latitudes"]))


def test_mirror_is_an_isometry_and_keeps_sin_latitude():
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(0, 360, 200), rng.uniform(-80, -20, 200)
    x = lonlat_to_stereo(lon, -lat)  # 12_dataset.py: southern parcels mirrored to the north
    s = sin_lat(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(s, np.sin(np.deg2rad(-lat)), atol=1e-9)
    lo, la = stereo_to_lonlat(x)
    np.testing.assert_allclose(la, -lat, atol=1e-9)
    # distances are kept: chord between two mirrored points equals the chord of the originals
    def chord(lo1, la1, lo2, la2):
        u = lambda o, a: np.stack([np.cos(np.deg2rad(a)) * np.cos(np.deg2rad(o)),
                                   np.cos(np.deg2rad(a)) * np.sin(np.deg2rad(o)), np.sin(np.deg2rad(a))], -1)
        return np.linalg.norm(u(lo1, la1) - u(lo2, la2), axis=-1)
    np.testing.assert_allclose(chord(lon[:-1], lat[:-1], lon[1:], lat[1:]), chord(lon[:-1], -lat[:-1], lon[1:], -lat[1:]))


@pytest.fixture(scope="module")
def fake_root():
    Path(BASE).mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="era5_test_", dir=BASE))
    (d / "data/tracks").mkdir(parents=True)
    (d / "configs").mkdir()
    shutil.copy(ROOT / "configs/releases.json", d / "configs/releases.json")
    rng = np.random.default_rng(1)
    T, levels, patches, n = 157, 2, [0, 6, 8, 14], 480
    N = levels * len(patches) * n
    lat0 = np.repeat(np.where(np.isin(patches, [8, 14]), -45.0, 45.0), n)
    lat0 = np.tile(lat0, levels) + rng.normal(0, 1, N)
    lon0 = rng.uniform(0, 360, N)
    t = np.arange(T, dtype=float)
    lon = lon0[None] + 0.2 * t[:, None]
    lat = lat0[None] + 0.0 * t[:, None]
    phi = np.abs(lat)
    x = np.concatenate([lonlat_to_stereo(lon, phi), np.full((T, N, 1), 5.0)], -1).astype(np.float32)
    for k, name in enumerate(["tr000", "tr001"]):
        drop = slice(0, 5) if k == 1 else slice(0, 0)  # release tr001 lost 5 parcels of the first patch
        keep = np.ones(N, bool)
        keep[drop] = False
        np.savez(d / f"data/tracks/tracks_{name}.npz", time_h=t, x_km=x[:, keep], v_kmh=np.gradient(x, axis=0)[:, keep],
                 a_kmh2=np.zeros_like(x)[:, keep], lat=lat[:, keep].astype(np.float32),
                 p_hpa=np.full((T, keep.sum()), 500.0, np.float32), level=np.repeat(np.arange(levels), len(patches) * n)[keep],
                 patch=np.tile(np.repeat(patches, n), levels)[keep], hemisphere=np.where(lat0 >= 0, 1, -1)[keep].astype(np.int8),
                 sigma=np.array([0.967, 0.93]), release_time="2015-01-01T00", dropped=int((~keep).sum()))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_load_patches_and_disjoint_sampling(fake_root):
    D = load_patches(fake_root, ["tr000", "tr001"], [0, 8], device="cpu", extra=True)
    assert D.X.shape == (2 * 2 * 2, 157, 480, 3) and D.filled == 5
    assert {m[3] for m in D.meta} == {1, -1}
    s = sin_lat(D.X)
    np.testing.assert_allclose(s.numpy(), np.sin(np.deg2rad(np.abs(D.LAT.numpy()))), atol=1e-5)
    g = torch.Generator().manual_seed(0)
    tok, qry, acc, score = sample_batch(D, 4, K=192, Q=256, gen=g)
    assert tok.shape == (4, 17 * 192, 8) and qry.shape == (4, 256, 8) and acc.shape == score.shape == (4, 256, 3)
    with pytest.raises(AssertionError):
        sample_batch(D, 2, K=300, Q=256, gen=g)
