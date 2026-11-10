"""Export the conditional Kinematic Interpolant to ONNX for in-browser inference (onnxruntime-web).

Two graphs: encoder (context tokens (1, 3264, 8) -> latents (1, 96, d)) and decoder (latents, queries (1, Q, 8) ->
acceleration and scaled score (1, Q, 3) each; Q dynamic). Checked against PyTorch on random inputs. An 8-bit copy of
each graph quantizes only the weights of the matrix products, per output channel (dynamic quantization of all
operators with one scale per tensor breaks the encoder); 06_check_onnx.py checks it on real tracks and the page uses
it (40 MB instead of 129 MB, about twice as fast in onnxruntime-web). Writes hf_model/onnx/{encoder,decoder}.onnx,
{encoder,decoder}.int8.onnx and config.json (scales, ensemble noise, training noise, context layout, the page's
settings: parcels launched, members, steps per hour).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "space"))
from ctx_model import ContextKI  # noqa: E402

CK = ROOT.parent / "03_era5/outputs/models/context_ki_era5.pt"
OUT = ROOT / "hf_model/onnx"


class Enc(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, tokens):
        return self.m.encode(tokens)


class Dec(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, z, queries):
        a, s = self.m.decode(z, queries)
        return a, s


if __name__ == "__main__":
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    m = ContextKI(d=ck["args"]["d_model"]).eval()
    m.load_state_dict(ck["state"])
    tok = torch.randn(1, 17 * 192, 8)
    with torch.no_grad():
        z = m.encode(tok)
        q = torch.randn(1, 256, 8)
        a, s = m.decode(z, q)
    torch.onnx.export(Enc(m), (tok,), OUT / "encoder.onnx", input_names=["tokens"], output_names=["latents"],
                      opset_version=17, dynamo=False)
    torch.onnx.export(Dec(m), (z, q), OUT / "decoder.onnx", input_names=["latents", "queries"],
                      output_names=["acceleration", "score"], dynamic_axes={"queries": {1: "Q"}, "acceleration": {1: "Q"},
                                                                              "score": {1: "Q"}},
                      opset_version=17, dynamo=False)
    for name in ("encoder", "decoder"):
        quantize_dynamic(OUT / f"{name}.onnx", OUT / f"{name}.int8.onnx", weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul", "Gemm"])
    for suf in ("", ".int8"):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        e = ort.InferenceSession(str(OUT / f"encoder{suf}.onnx"), so, providers=["CPUExecutionProvider"])
        d = ort.InferenceSession(str(OUT / f"decoder{suf}.onnx"), so, providers=["CPUExecutionProvider"])
        zo = e.run(None, {"tokens": tok.numpy()})[0]
        ao, so_ = d.run(None, {"latents": zo, "queries": q.numpy()})
        print(f"{'fp32' if not suf else 'int8'}: latents max |diff| {np.abs(zo - z.numpy()).max():.2e}, "
              f"acceleration max |diff| {np.abs(ao - a.numpy()).max():.2e} (|a| max {np.abs(a.numpy()).max():.2f}), "
              f"score max |diff| {np.abs(so_ - s.numpy()).max():.2e}; sizes "
              f"{(OUT / f'encoder{suf}.onnx').stat().st_size / 1e6:.1f} + {(OUT / f'decoder{suf}.onnx').stat().st_size / 1e6:.1f} MB")
    ens = json.loads((ROOT.parent / "04_rwb_forecast/configs/ensemble.json").read_text())
    (OUT / "config.json").write_text(json.dumps(dict(
        scales=ck["scales"], eps_rel=ens["eps_rel"], members=ens["members"], sx_km=[5.0, 5.0, 0.01], delta_h=6.0,
        omega=float(np.pi * np.sqrt(3)), context_parcels=192, context_step_h=3, context_span_h=48, steps_per_h=6,
        launch_parcels=16, graphs=dict(encoder="encoder.int8.onnx", decoder="decoder.int8.onnx"),
        h_km=7.0, a_km=6371.0, d_model=ck["args"]["d_model"]), indent=1))
    print("wrote", sorted(p.name for p in OUT.iterdir()))
