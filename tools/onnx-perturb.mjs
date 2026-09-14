// Perturb ONE intermediate tensor by a relative amount and re-run the model through onnxruntime.
//
// 🔴 WHY. "Our value differs from onnxruntime's by 1.4 ulp" is only half an answer - the other half is what
// 1.4 ulp DOES to the output. A graph that takes sin() of a 98,919-radian phase has no bits left to spare
// there, and if perturbing onnxruntime's OWN phase by our exact error reproduces our exact final
// correlation, then that correlation is the floor for any float32 engine and there is no bug left to find.
// Without this test, "it's just precision" is a guess.
//
// Splices Mul(tensor, scale) in front of every consumer of the named tensor, where `scale` is a tensor of
// 1+eps drawn deterministically, then writes the resulting waveform.
//
//   node tools/onnx-perturb.mjs <model.onnx> <voice.bin> <tokens-csv> <outdir> <tensor> <relative-eps>
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-node';

const [modelPath, voicePath, tokensCsv, outDir, target, epsStr, dumpCsv] = process.argv.slice(2);
const eps = parseFloat(epsStr);
mkdirSync(outDir, { recursive: true });

const model = onnx.ModelProto.decode(readFileSync(modelPath));
const g = model.graph;

const producer = g.node.find(n => n.output.includes(target));
if (!producer) { console.log(`SKIP ${target} (not produced by any node)`); process.exit(1); }

// The perturbation is a full-size tensor, not a scalar: a scalar scale is a systematic gain, which is
// exactly the error mode float rounding does NOT have. Deterministic so the run is repeatable.
// Its length must match the target's, which is read from a first pass that just dumps the tensor.
const probe = { ...model };
const shapeRun = await (async () => {
  const m2 = onnx.ModelProto.decode(readFileSync(modelPath));
  m2.graph.output.push(onnx.ValueInfoProto.create({
    name: target, type: { tensorType: { elemType: onnx.TensorProto.DataType.FLOAT } },
  }));
  const p = join(outDir, 'probe.onnx');
  writeFileSync(p, onnx.ModelProto.encode(m2).finish());
  const s = await ort.InferenceSession.create(p);
  const r = await s.run(feeds());
  return r[target];
})();
const count = shapeRun.data.length;
console.log(`target ${target} dims=[${shapeRun.dims.join(',')}] n=${count}`);

let seed = 12345 >>> 0;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

// ⚠️ ADDITIVE, not multiplicative. `relRMS` is ||ours-ref|| / ||ref|| - a whole-tensor additive measure -
// and a multiplicative (1+eps) perturbation is a different error entirely: it can never change a value's
// SIGN, so on a tensor whose interesting behaviour is a division by something near zero it reproduces
// nothing. MEASURED: a 1.8e-3 multiplicative perturbation of this model's STFT output changed the waveform
// not at all (corr 1.000000), because every near-zero real part stayed on its own side of zero.
const ref = Float32Array.from(shapeRun.data);
let rms = 0; for (const v of ref) rms += v * v;
rms = Math.sqrt(rms / count);
const delta = new Float32Array(count);
// Uniform in [-a, a] has RMS a/sqrt(3); scale so the injected RMS is exactly eps * rms(tensor).
const amp = eps * rms * Math.sqrt(3);
for (let i = 0; i < count; i++) delta[i] = (rnd() * 2 - 1) * amp;

g.initializer.push(onnx.TensorProto.create({
  name: '__perturb_delta', dataType: onnx.TensorProto.DataType.FLOAT, dims: shapeRun.dims,
  rawData: new Uint8Array(delta.buffer.slice(0)),
}));
const perturbed = target + '__perturbed';
for (const n of g.node) {
  for (let i = 0; i < n.input.length; i++) if (n.input[i] === target) n.input[i] = perturbed;
}
g.node.push(onnx.NodeProto.create({
  opType: 'Add', input: [target, '__perturb_delta'], output: [perturbed], name: '__perturb',
}));

// Optional: also expose intermediates AFTER the perturbation, so the question "does onnxruntime's own
// Atan degrade this much when its input carries our error?" can be answered directly instead of inferred.
const dumps = (dumpCsv ?? '').split(',').map(s2 => s2.trim()).filter(Boolean);
const produced2 = new Set(g.node.flatMap(n => n.output));
const added2 = [];
for (const name of dumps) {
  if (!produced2.has(name)) { console.log(`SKIP ${name} (not produced)`); continue; }
  g.output.push(onnx.ValueInfoProto.create({
    name, type: { tensorType: { elemType: onnx.TensorProto.DataType.FLOAT } },
  }));
  added2.push(name);
}

const outPath = join(outDir, 'perturbed.onnx');
writeFileSync(outPath, onnx.ModelProto.encode(model).finish());
const session = await ort.InferenceSession.create(outPath);
const results = await session.run(feeds());
for (const name of added2) {
  const t = results[name];
  if (!t) { console.log(`MISSING ${name}`); continue; }
  const d2 = Float32Array.from(t.data);
  writeFileSync(join(outDir, name.replace(/[^A-Za-z0-9._-]/g, '_') + '.f32'), Buffer.from(d2.buffer));
  console.log(`PERT ${name} dims=[${t.dims.join(',')}] n=${d2.length}`);
}
const wav = Float32Array.from(results[session.outputNames[0]].data);
writeFileSync(join(outDir, 'waveform.f32'), Buffer.from(wav.buffer));
let peak = 0; for (const v of wav) peak = Math.max(peak, Math.abs(v));
console.log(`perturbed ${target} by relRMS ${eps} -> waveform n=${wav.length} peak=${peak.toFixed(5)}`);

function feeds() {
  const tokens = tokensCsv.split(',').map(s => BigInt(s.trim()));
  const voiceRaw = readFileSync(voicePath);
  const DIM = 256, ROWS = 510;
  const row = Math.min(Math.max(tokens.length, 0), ROWS - 1);
  const style = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) style[i] = voiceRaw.readFloatLE((row * DIM + i) * 4);
  return {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(tokens), [1, tokens.length]),
    style: new ort.Tensor('float32', style, [1, DIM]),
    speed: new ort.Tensor('float32', Float32Array.from([1.0]), [1]),
  };
}
