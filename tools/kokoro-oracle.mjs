// Kokoro reference output via onnxruntime-node - the oracle our engine is compared against.
//
// 🔴 WHY. Our engine now runs the whole 2,323-node graph and produces audio whose PEAK IS 2226, where a
// waveform must be within [-1, 1]. Shapes are right and values are not, and no amount of reading the
// graph settles which node first diverges - only a reference does. ORT runs the identical file with the
// identical inputs, so any difference is ours.
//
// Writes the reference waveform as raw float32 so the C# side can diff it sample by sample.
//
//   node tools/kokoro-oracle.mjs <model.onnx> <voice.bin> <tokens-csv> <out.f32>
import * as ort from 'onnxruntime-node';
import { readFileSync, writeFileSync } from 'node:fs';

const [modelPath, voicePath, tokensCsv, outPath] = process.argv.slice(2);
if (!modelPath || !voicePath || !tokensCsv || !outPath) {
  console.error('usage: node kokoro-oracle.mjs <model.onnx> <voice.bin> <tokens-csv> <out.f32>');
  process.exit(2);
}

const tokens = tokensCsv.split(',').map(s => BigInt(s.trim()));
// The style row is chosen by TOKEN COUNT - the voice file is a 510x256 table, not a vector.
const voiceRaw = readFileSync(voicePath);
const DIM = 256, ROWS = 510;
const row = Math.min(Math.max(tokens.length, 0), ROWS - 1);
const style = new Float32Array(DIM);
for (let i = 0; i < DIM; i++) style[i] = voiceRaw.readFloatLE((row * DIM + i) * 4);

const session = await ort.InferenceSession.create(modelPath);
console.log('inputs :', session.inputNames.join(', '));
console.log('outputs:', session.outputNames.join(', '));

const feeds = {};
for (const name of session.inputNames) {
  if (name === 'input_ids' || name === 'tokens')
    feeds[name] = new ort.Tensor('int64', BigInt64Array.from(tokens), [1, tokens.length]);
  else if (name === 'style' || name === 'ref_s')
    feeds[name] = new ort.Tensor('float32', style, [1, DIM]);
  else if (name === 'speed')
    feeds[name] = new ort.Tensor('float32', Float32Array.from([1.0]), [1]);
}

const results = await session.run(feeds);
const out = results[session.outputNames[0]];
const data = Float32Array.from(out.data);
let peak = 0;
for (const v of data) { const a = Math.abs(v); if (a > peak) peak = a; }
console.log(`reference: ${data.length} samples, dims [${out.dims.join(',')}], peak ${peak.toFixed(4)}`);
writeFileSync(outPath, Buffer.from(data.buffer));
console.log(`wrote ${outPath}`);
