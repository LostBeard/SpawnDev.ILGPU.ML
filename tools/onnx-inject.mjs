// Replace ONE intermediate tensor with values from a file and re-run the model through onnxruntime.
//
// 🔴 WHY. Node-level diffs answer "where do we first differ"; they cannot answer "is the rest of our graph
// CORRECT given what we feed it". When a graph is ill-conditioned - this one takes sin() of a 98,919-radian
// phase - every node after the sensitive point differs from the reference no matter how right the code is,
// and statistics alone can never separate that from a real defect.
//
// Splicing OUR tensor into the reference engine removes the conditioning from the question entirely: run
// onnxruntime from our value onward, and anything that still differs is OUR bug. Anything that now matches
// was never broken.
//
//   node tools/onnx-inject.mjs <model.onnx> <voice.bin> <tokens-csv> <outdir> <tensor> <values.f32> [dump,names]
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-node';

const [modelPath, voicePath, tokensCsv, outDir, target, valuePath, dumpCsv] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });

const model = onnx.ModelProto.decode(readFileSync(modelPath));
const g = model.graph;
if (!g.node.some(n => n.output.includes(target))) {
  console.log(`SKIP ${target} (not produced by any node)`); process.exit(1);
}

// The shape has to come from a real run: an injected tensor with the right COUNT but the wrong shape is
// silently a different computation, and onnxruntime will happily broadcast it.
const shapeRun = await (async () => {
  const m2 = onnx.ModelProto.decode(readFileSync(modelPath));
  m2.graph.output.push(onnx.ValueInfoProto.create({
    name: target, type: { tensorType: { elemType: onnx.TensorProto.DataType.FLOAT } },
  }));
  const p = join(outDir, 'probe.onnx');
  writeFileSync(p, onnx.ModelProto.encode(m2).finish());
  const s = await ort.InferenceSession.create(p);
  return (await s.run(feeds()))[target];
})();

const raw = readFileSync(valuePath);
const values = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
if (values.length !== shapeRun.data.length) {
  console.log(`MISMATCH ${target}: file has ${values.length}, graph wants ${shapeRun.data.length}`);
  process.exit(1);
}
let dd = 0, rr = 0;
for (let i = 0; i < values.length; i++) { const d = values[i] - shapeRun.data[i]; dd += d * d; rr += shapeRun.data[i] ** 2; }
console.log(`inject ${target} dims=[${shapeRun.dims.join(',')}] n=${values.length} `
          + `relRMS(injected vs native)=${Math.sqrt(dd / rr).toExponential(2)}`);

g.initializer.push(onnx.TensorProto.create({
  name: '__injected', dataType: onnx.TensorProto.DataType.FLOAT, dims: shapeRun.dims,
  rawData: new Uint8Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.length * 4)),
}));
// Rewire consumers only - the producing node stays, dead but harmless, so node indices and every other
// consumer's view of the graph are untouched.
for (const n of g.node)
  for (let i = 0; i < n.input.length; i++) if (n.input[i] === target) n.input[i] = '__injected';

const dumps = (dumpCsv ?? '').split(',').map(s => s.trim()).filter(Boolean);
const produced = new Set(g.node.flatMap(n => n.output));
const added = [];
for (const name of dumps) {
  if (!produced.has(name)) { console.log(`SKIP ${name} (not produced)`); continue; }
  g.output.push(onnx.ValueInfoProto.create({
    name, type: { tensorType: { elemType: onnx.TensorProto.DataType.FLOAT } },
  }));
  added.push(name);
}

const outPath = join(outDir, 'injected.onnx');
writeFileSync(outPath, onnx.ModelProto.encode(model).finish());
const session = await ort.InferenceSession.create(outPath);
const results = await session.run(feeds());
for (const name of added) {
  const t = results[name];
  if (!t) { console.log(`MISSING ${name}`); continue; }
  const d = Float32Array.from(t.data);
  writeFileSync(join(outDir, name.replace(/[^A-Za-z0-9._-]/g, '_') + '.f32'), Buffer.from(d.buffer));
  console.log(`INJ ${name} dims=[${t.dims.join(',')}] n=${d.length}`);
}
const wav = Float32Array.from(results[session.outputNames[0]].data);
writeFileSync(join(outDir, 'waveform.f32'), Buffer.from(wav.buffer));
let peak = 0; for (const v of wav) peak = Math.max(peak, Math.abs(v));
console.log(`waveform n=${wav.length} peak=${peak.toFixed(5)}`);

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
