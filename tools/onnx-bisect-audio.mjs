// Cut an ONNX audio-front-end graph at each named tensor, run every cut through onnxruntime on a WAV, and save
// the cut model plus ORT's value - the other half of tools/parakeet-harness's `--probe-dir`, which runs the SAME
// cut through our engine and prints the FIRST tensor that disagrees.
//
//   node tools/onnx-bisect-audio.mjs <model.onnx> <audio.wav 16k mono> <outDir> <tensor1,tensor2,...>
//
// WHY: comparing a graph's final output says THAT it diverged, never WHERE. Each cut makes one tensor the graph's
// only output, so "the mel is wrong" becomes "STFT is right and the MatMul after it is the first that is not".
// Inputs are bound by name: `waveforms` = the WAV as [1,N] float32, `waveforms_lens` = [N] int64 (the NeMo
// preprocessor signature). A cut that ORT fused away is reported, not hidden: ask for its producer's input.
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import * as ort from 'onnxruntime-node';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const [modelPath, wavPath, outDir, namesCsv] = process.argv.slice(2);
if (!namesCsv) { console.error('usage: node tools/onnx-bisect-audio.mjs <model.onnx> <wav> <outDir> <names,...>'); process.exit(2); }
mkdirSync(outDir, { recursive: true });

function wav16kMono(path) {
  const b = readFileSync(path);
  let p = 12, fmt = null, data = null;
  while (p + 8 <= b.length) {
    const id = b.toString('ascii', p, p + 4), size = b.readUInt32LE(p + 4);
    if (id === 'fmt ') fmt = { rate: b.readUInt32LE(p + 12), ch: b.readUInt16LE(p + 10), bits: b.readUInt16LE(p + 22) };
    if (id === 'data') { data = b.subarray(p + 8, Math.min(b.length, p + 8 + size)); break; }
    p += 8 + size + (size & 1);
  }
  if (!fmt || fmt.rate !== 16000 || fmt.ch !== 1 || fmt.bits !== 16) throw new Error('need 16 kHz mono 16-bit PCM');
  const out = new Float32Array(data.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = data.readInt16LE(i * 2) / 32768;
  return out;
}

const samples = wav16kMono(wavPath);
const base = onnx.ModelProto.decode(readFileSync(modelPath));
const produced = new Map();
for (const n of base.graph.node) for (const o of n.output) produced.set(o, n);

// A cut may be a TENSOR name, or `@<suffix>`: the first output of the node whose NAME ends with <suffix> (the
// last such node when several do). The tensors actually cut are written to <outDir>/cuts.txt for the harness.
const resolveCut = c => {
  if (!c.startsWith('@')) return c;
  const suffix = c.slice(1);
  const hits = base.graph.node.filter(n => (n.name || '').endsWith(suffix));
  if (hits.length === 0) { console.log(`SKIP ${c}: no node name ends with it`); return null; }
  return hits[hits.length - 1].output[0];
};
const cutNames = namesCsv.split(',').map(s => s.trim()).filter(Boolean).map(resolveCut).filter(Boolean);
writeFileSync(join(outDir, 'cuts.txt'), cutNames.join(','));
for (const name of cutNames) {
  if (!produced.has(name) && !base.graph.output.some(o => o.name === name)) { console.log(`SKIP ${name}: not produced by any node`); continue; }
  const cut = onnx.ModelProto.decode(readFileSync(modelPath));
  // Value info is unknown for an intermediate; an empty ValueInfo (name only) is legal for an output.
  cut.graph.output = [onnx.ValueInfoProto.create({ name })];
  const bytes = onnx.ModelProto.encode(cut).finish();
  const file = name.replaceAll('/', '__');                       // tensor names may be paths
  writeFileSync(join(outDir, `${file}.onnx`), bytes);
  try {
    const session = await ort.InferenceSession.create(bytes, { graphOptimizationLevel: 'disabled', executionProviders: ['cpu'] });
    const feeds = {};
    for (const input of session.inputNames) {
      if (input === 'audio') {                                        // speaker encoders: fixed 6 s window, repeat-fill / centre-crop
        const W = 96000, win = new Float32Array(W);
        if (samples.length >= W) { const off = Math.floor((samples.length - W) / 2); win.set(samples.subarray(off, off + W)); }
        else for (let i = 0; i < W; i++) win[i] = samples[i % samples.length];
        feeds[input] = new ort.Tensor('float32', win, [1, W]);
      }
      else if (input === 'waveforms') feeds[input] = new ort.Tensor('float32', samples, [1, samples.length]);
      else if (input === 'waveforms_lens') feeds[input] = new ort.Tensor('int64', BigInt64Array.from([BigInt(samples.length)]), [1]);
      else throw new Error(`no binding for input '${input}'`);
    }
    const t = (await session.run(feeds))[name];
    const f32 = Float32Array.from(t.data, Number);
    writeFileSync(join(outDir, `${file}.f32`), Buffer.from(f32.buffer));
    console.log(`${name.padEnd(28)} dims [${t.dims.join(',')}] ${t.type}  first ${Array.from(f32.slice(0, 4)).map(v => v.toPrecision(5)).join(' ')}`);
  } catch (e) { console.log(`FAIL ${name}: ${e.message}`); }
}
