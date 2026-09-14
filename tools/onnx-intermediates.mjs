// Run a model through onnxruntime and dump INTERMEDIATE tensors, by adding them as graph outputs.
//
// 🔴 WHY. Comparing final outputs tells you THAT a 2,323-node graph diverged, never WHERE. ORT will only
// hand back tensors listed in graph.output - so this rewrites the model to list the ones asked for. That
// turns "the audio is wrong" into "node N is the first that disagrees", which is the difference between
// bisecting and guessing.
//
// ⚠️ Adding an output can inhibit ORT's graph optimisations for that subgraph, so ask for the tensors you
// need rather than hundreds at once, and treat a tensor that vanishes (fused away) as "ask for its
// producer's input instead", not as a failure.
//
//   node tools/onnx-intermediates.mjs <model.onnx> <voice.bin> <tokens-csv> <outdir> <name1,name2,...>
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-node';

const [modelPath, voicePath, tokensCsv, outDir, namesCsv] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const wanted = namesCsv.split(',').map(s => s.trim()).filter(Boolean);

const model = onnx.ModelProto.decode(readFileSync(modelPath));
const existing = new Set(model.graph.output.map(o => o.name));
// Every tensor produced anywhere in the graph, so a typo is reported here rather than as an ORT error.
const produced = new Set(model.graph.node.flatMap(n => n.output));
const added = [];
for (const name of wanted) {
  if (existing.has(name)) continue;
  if (!produced.has(name)) { console.log(`SKIP  ${name} (not produced by any node)`); continue; }
  model.graph.output.push(onnx.ValueInfoProto.create({
    name,
    // No type: ORT infers it. Declaring a wrong one is worse than declaring none.
    type: { tensorType: { elemType: onnx.TensorProto.DataType.FLOAT } },
  }));
  added.push(name);
}
const patched = join(outDir, 'patched.onnx');
writeFileSync(patched, onnx.ModelProto.encode(model).finish());
console.log(`patched model: +${added.length} outputs`);

const tokens = tokensCsv.split(',').map(s => BigInt(s.trim()));
const voiceRaw = readFileSync(voicePath);
const DIM = 256, ROWS = 510;
const row = Math.min(Math.max(tokens.length, 0), ROWS - 1);
const style = new Float32Array(DIM);
for (let i = 0; i < DIM; i++) style[i] = voiceRaw.readFloatLE((row * DIM + i) * 4);

const session = await ort.InferenceSession.create(patched);
const feeds = {
  input_ids: new ort.Tensor('int64', BigInt64Array.from(tokens), [1, tokens.length]),
  style: new ort.Tensor('float32', style, [1, DIM]),
  speed: new ort.Tensor('float32', Float32Array.from([1.0]), [1]),
};
const results = await session.run(feeds);

for (const name of [...added, session.outputNames[0]]) {
  const t = results[name];
  if (!t) { console.log(`MISSING ${name}`); continue; }
  const data = Float32Array.from(t.data);
  let peak = 0;
  for (const v of data) { const a = Math.abs(v); if (a > peak) peak = a; }
  const safe = name.replace(/[^A-Za-z0-9._-]/g, '_');
  writeFileSync(join(outDir, `${safe}.f32`), Buffer.from(data.buffer));
  console.log(`REF ${name} dims=[${t.dims.join(',')}] n=${data.length} peak=${peak.toFixed(5)} `
            + `first8=${Array.from(data.slice(0, 8)).map(v => v.toFixed(5)).join(' ')}`);
}
