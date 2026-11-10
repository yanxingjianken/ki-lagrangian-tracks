// Time the ONNX graphs in onnxruntime-web (WebAssembly, one thread, as on a page without cross-origin isolation).
// Usage: node scripts/07_wasm_timing.mjs <path to node_modules/onnxruntime-web> ; writes outputs/wasm_timing.json
// (milliseconds per encoder call and per decoder call for several numbers of queries) for the graphs named in
// hf_model/onnx/config.json.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ort = await import(path.join(process.argv[2], "dist/ort.wasm.min.mjs"));
ort.env.wasm.numThreads = 1;
const dir = path.join(root, "hf_model/onnx");
const cfg = JSON.parse(readFileSync(path.join(dir, "config.json"), "utf8"));      // the graphs the page uses
const enc = await ort.InferenceSession.create(readFileSync(path.join(dir, cfg.graphs.encoder)), { executionProviders: ["wasm"] });
const dec = await ort.InferenceSession.create(readFileSync(path.join(dir, cfg.graphs.decoder)), { executionProviders: ["wasm"] });
const tok = new ort.Tensor("float32", Float32Array.from({ length: 3264 * 8 }, () => Math.random() - 0.5), [1, 3264, 8]);
let t = performance.now();
const z = (await enc.run({ tokens: tok })).latents;
const out = { version: "onnxruntime-web 1.30.0, wasm, 1 thread", graphs: cfg.graphs, encoder_ms: performance.now() - t, decoder_ms: {} };
for (const Q of [64, 128, 256]) {
  const q = new ort.Tensor("float32", Float32Array.from({ length: Q * 8 }, () => Math.random() - 0.5), [1, Q, 8]);
  await dec.run({ latents: z, queries: q });
  t = performance.now();
  for (let i = 0; i < 10; i++) await dec.run({ latents: z, queries: q });
  out.decoder_ms[Q] = (performance.now() - t) / 10;
}
mkdirSync(path.join(root, "outputs"), { recursive: true });
writeFileSync(path.join(root, "outputs/wasm_timing.json"), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
