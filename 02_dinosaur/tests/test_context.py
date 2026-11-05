import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ctx_data import context, rollout, sample_batch  # noqa: E402
from ctx_model import ContextKI  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def toy(Np=3, T=60, N=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.cumsum(torch.randn(Np, T, N, 3, generator=g) * torch.tensor([30.0, 30.0, 0.05]), 1)
    X = X + torch.tensor([3000.0, -2000.0, 5.0])
    return SimpleNamespace(X=X.to(DEV), V=(torch.randn(Np, T, N, 3, generator=g) * torch.tensor([50.0, 50.0, 0.2])).to(DEV),
                           A=(torch.randn(Np, T, N, 3, generator=g) * torch.tensor([5.0, 5.0, 0.02])).to(DEV),
                           S=torch.full((Np, T, N), 0.7, device=DEV))


def test_model_shapes():
    m = ContextKI(d=64, latents=8, enc_layers=1, dec_layers=1).to(DEV)
    a, s = m(torch.randn(2, 30, 8, device=DEV), None, torch.randn(2, 7, 8, device=DEV))
    assert a.shape == (2, 7, 3) and s.shape == (2, 7, 3)


class Straight(torch.nn.Module):
    def encode(self, tok):
        return None

    def decode(self, z, q):
        return torch.zeros_like(q[..., 1:4]), None


@pytest.mark.parametrize("direction", [1, -1])
def test_rollout_straight_motion_is_exact(direction):
    x0 = torch.tensor([[[1000.0, -500.0, 3.0]]], device=DEV)
    v0 = torch.tensor([[[40.0, 10.0, 0.1]]], device=DEV)
    path = rollout(Straight(), torch.zeros(1, 1, 8, device=DEV), torch.zeros(1, 2, device=DEV), x0, v0,
                   torch.tensor([direction], device=DEV), horizon_h=12)
    assert torch.allclose(path[-1], x0 + direction * 12 * v0, atol=1e-3)


def test_context_and_batch_shapes():
    D = toy()
    p, t0 = torch.tensor([0, 1], device=DEV), torch.tensor([30, 30], device=DEV)
    tok, cen = context(D, p, t0, torch.tensor([1, -1], device=DEV), torch.arange(5, device=DEV).repeat(2, 1), 12, 3)
    assert tok.shape == (2, 25, 8) and (tok[0, :, 0] <= 0).all() and (tok[1, :, 0] >= 0).all()
    assert torch.allclose(tok[0, :, 3] * 10.0, D.X[0, torch.arange(30, 17, -3, device=DEV)][:, :5, 2].flatten(), atol=1e-4)
    tok, qry, acc, score = sample_batch(D, 4, K=5, Q=16, span_h=12, horizon_h=6,
                                        gen=torch.Generator(device=DEV).manual_seed(1))
    assert qry.shape == (4, 16, 8) and acc.shape == (4, 16, 3) and score.shape == (4, 16, 3)
    assert torch.isfinite(acc).all()


def test_flowfields_on_solid_body_rotation():
    import numpy as np
    from flowfields import A_M, mirror_global, streamfunction, vorticity
    lat = np.linspace(0, 90, 91)
    u = np.repeat((20.0 * np.cos(np.deg2rad(lat)))[:, None], 360, 1)[None]
    ug, vg = mirror_global(u, np.zeros_like(u), lat)
    latg = np.linspace(90, -90, 181)
    zeta, psi = vorticity(ug, vg)[0], streamfunction(ug, vg)[0]
    expect = 2 * 20.0 * np.sin(np.deg2rad(latg)) / A_M
    assert np.abs(zeta - expect[:, None]).max() < 1e-3 * np.abs(expect).max()
    u_psi = -np.gradient(psi, np.deg2rad(latg), axis=0) / A_M
    assert np.abs(u_psi[30:151] - ug[0, 30:151]).max() < 0.05
