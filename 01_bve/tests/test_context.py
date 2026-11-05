import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ctx_data import OMEGA, context, rollout, sample_batch  # noqa: E402
from ctx_model import ACC_KMH2, POS_KM, VEL_KMH, ContextKI  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def toy(Np=3, T=60, N=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.cumsum(torch.randn(Np, T, N, 2, generator=g) * 30, 1) + 3000
    return SimpleNamespace(X=X.to(DEV), V=torch.randn(Np, T, N, 2, generator=g).to(DEV) * 50,
                           A=torch.randn(Np, T, N, 2, generator=g).to(DEV) * 5,
                           S=torch.full((Np, T, N), 0.7, device=DEV))


def test_model_shapes():
    m = ContextKI(d=64, latents=8, enc_layers=1, dec_layers=1).to(DEV)
    a, s = m(torch.randn(2, 30, 6, device=DEV), None, torch.randn(2, 7, 6, device=DEV))
    assert a.shape == (2, 7, 2) and s.shape == (2, 7, 2)


class Straight(torch.nn.Module):
    def encode(self, tok):
        return None

    def decode(self, z, q):
        return torch.zeros_like(q[..., 1:3]), None


@pytest.mark.parametrize("direction", [1, -1])
def test_rollout_straight_motion_is_exact(direction):
    x0 = torch.tensor([[[1000.0, -500.0]]], device=DEV)
    v0 = torch.tensor([[[40.0, 10.0]]], device=DEV)
    path = rollout(Straight(), torch.zeros(1, 1, 6, device=DEV), torch.zeros(1, 2, device=DEV), x0, v0,
                   torch.tensor([direction], device=DEV), horizon_h=12)
    assert torch.allclose(path[-1], x0 + direction * 12 * v0, atol=1e-3)


def test_context_tokens_are_on_the_observed_side():
    D = toy()
    p, t0 = torch.tensor([0, 1], device=DEV), torch.tensor([30, 30], device=DEV)
    tok, cen = context(D, p, t0, torch.tensor([1, -1], device=DEV), torch.arange(5, device=DEV).repeat(2, 1), 12, 3)
    assert (tok[0, :, 0] <= 0).all() and (tok[1, :, 0] >= 0).all()
    assert torch.allclose(cen, D.X[p, t0].mean(1))


def test_noise_restoring_term():
    D = toy()
    tok, qry, acc, score = sample_batch(D, 4, K=5, Q=16, span_h=12, horizon_h=6, sx_km=5.0, delta_h=6.0,
                                        gen=torch.Generator(device=DEV).manual_seed(1))
    assert tok.shape == (4, 5 * 5, 6) and qry.shape == (4, 16, 6)
    assert acc.shape == (4, 16, 2) and score.shape == (4, 16, 2)
    assert torch.isfinite(acc).all() and (qry[..., 0].abs() <= 6 / 24 + 1e-6).all()
    assert abs(float(score.pow(2).mean()) - 1.0) < 0.3


@pytest.mark.skipif(not (ROOT / "data/tracks_lagranto_j1_m4_w0.npz").exists(), reason="tracks not built")
def test_flow_set_tracks_complete():
    flows = json.loads((ROOT / "configs/flows.json").read_text())["flows"]
    for f in flows:
        for w in range(4):
            with np.load(ROOT / f"data/tracks_lagranto_{f['name']}_w{w}.npz") as d:
                assert d["x_km"].shape == (157, 20000, 2)
                assert np.bincount(d["patch"]).tolist() == [1000] * 20
                assert np.isfinite(d["a_kmh2"]).all()
                assert np.allclose(d["time_h"], np.arange(157))
