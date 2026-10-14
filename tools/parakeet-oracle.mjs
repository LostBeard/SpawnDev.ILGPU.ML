// Parakeet TDT 0.6B v3 on onnxruntime-node: the REFERENCE our ParakeetTdtPipeline is held to.
//
//   node tools/parakeet-oracle.mjs <dir with nemo128/encoder/decoder_joint/vocab> <audio.wav 16k mono> [outDir]
//
// WHY: a transcript from our engine proves nothing until it is compared with what the reference runtime
// makes of the SAME bytes. This runs the three graphs exactly as istupakov/onnx-asr does (greedy TDT: one
// joint evaluation per frame, token + duration argmax, duration > 0 skips frames) and, with outDir, dumps the
// mel features and encoder output as raw float32 so a divergence is localised to a stage (mel / encoder /
// decode) instead of "the words are wrong".
import * as ort from 'onnxruntime-node';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const [dir, wavPath, outDir] = process.argv.slice(2);
if (!dir || !wavPath) { console.error('usage: node tools/parakeet-oracle.mjs <modelDir> <wav> [outDir]'); process.exit(2); }

function readWav16kMono(path) {
  const b = readFileSync(path);
  if (b.toString('ascii', 0, 4) !== 'RIFF') throw new Error('not a RIFF wav');
  let p = 12, fmt = null, data = null;
  while (p + 8 <= b.length) {
    const id = b.toString('ascii', p, p + 4), size = b.readUInt32LE(p + 4);
    if (id === 'fmt ') fmt = { tag: b.readUInt16LE(p + 8), ch: b.readUInt16LE(p + 10), rate: b.readUInt32LE(p + 12), bits: b.readUInt16LE(p + 22) };
    if (id === 'data') { data = b.subarray(p + 8, Math.min(b.length, p + 8 + size)); break; }
    p += 8 + size + (size & 1);
  }
  if (!fmt || !data) throw new Error('no fmt/data chunk');
  if (fmt.rate !== 16000 || fmt.ch !== 1) throw new Error(`need 16 kHz mono, got ${fmt.rate} Hz ${fmt.ch} ch`);
  const n = fmt.bits === 16 ? data.length / 2 : data.length / 4;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = fmt.bits === 16 ? data.readInt16LE(i * 2) / 32768 : (fmt.tag === 3 ? data.readFloatLE(i * 4) : data.readInt32LE(i * 4) / 2147483648);
  return out;
}

const vocab = readFileSync(join(dir, 'vocab.txt'), 'utf8').split('\n').filter(Boolean).map(l => { const i = l.lastIndexOf(' '); return [Number(l.slice(i + 1)), l.slice(0, i)]; });
const tokens = []; for (const [id, s] of vocab) tokens[id] = s;
const BLANK = tokens.length - 1;            // <blk> is the last entry (8192); the joint has BLANK+1 token logits then 5 duration logits
const VOCAB = tokens.length;                // 8193
const MAX_PER_FRAME = 10;

const samples = readWav16kMono(wavPath);
const t0 = performance.now();
const [pre, enc, dec] = await Promise.all(['nemo128.onnx', 'encoder-model.onnx', 'decoder_joint-model.onnx'].map(f => ort.InferenceSession.create(join(dir, f), { executionProviders: ['cpu'] })));
const tLoad = performance.now();

const feats = await pre.run({ waveforms: new ort.Tensor('float32', samples, [1, samples.length]), waveforms_lens: new ort.Tensor('int64', BigInt64Array.from([BigInt(samples.length)]), [1]) });
const tMel = performance.now();
const e = await enc.run({ audio_signal: feats.features, length: feats.features_lens });
const tEnc = performance.now();
const enc3 = e.outputs;                      // [1, 1024, T]
const T = Number(e.encoded_lengths.data[0]);
const D = enc3.dims[1], TT = enc3.dims[2];

let s1 = new ort.Tensor('float32', new Float32Array(2 * 640), [2, 1, 640]);
let s2 = new ort.Tensor('float32', new Float32Array(2 * 640), [2, 1, 640]);
let last = BLANK, t = 0, emitted = 0;
const out = [];                              // [token, startFrame]
const frame = new Float32Array(D);
while (t < T) {
  for (let d = 0; d < D; d++) frame[d] = enc3.data[d * TT + t];
  const r = await dec.run({
    encoder_outputs: new ort.Tensor('float32', frame, [1, D, 1]),
    targets: new ort.Tensor('int32', Int32Array.from([last]), [1, 1]),
    target_length: new ort.Tensor('int32', Int32Array.from([1]), [1]),
    input_states_1: s1, input_states_2: s2,
  });
  const lg = r.outputs.data;
  let tok = 0, dur = 0;
  for (let i = 1; i < VOCAB; i++) if (lg[i] > lg[tok]) tok = i;
  for (let i = 1; i < lg.length - VOCAB; i++) if (lg[VOCAB + i] > lg[VOCAB + dur]) dur = i;
  if (tok !== BLANK) { out.push([tok, t]); last = tok; s1 = r.output_states_1; s2 = r.output_states_2; }
  if (dur > 0) { t += dur; emitted = 0; }
  else if (tok === BLANK || ++emitted >= MAX_PER_FRAME) { t += 1; emitted = 0; }
}
const tDec = performance.now();

const text = out.map(([id]) => tokens[id]).join('').replaceAll('▁', ' ').trim();
console.log(`audio    : ${(samples.length / 16000).toFixed(2)} s`);
console.log(`frames   : mel ${feats.features.dims[2]}, encoder ${T}`);
console.log(`ms       : load ${(tLoad - t0).toFixed(0)}  mel ${(tMel - tLoad).toFixed(0)}  encoder ${(tEnc - tMel).toFixed(0)}  decode ${(tDec - tEnc).toFixed(0)} (CPU, onnxruntime-node)`);
console.log(`tokens   : ${out.map(([id]) => id).join(' ')}`);
console.log(`text     : ${text}`);

if (outDir) {
  mkdirSync(outDir, { recursive: true });
  const raw = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
  writeFileSync(join(outDir, 'features.f32'), raw(feats.features.data));
  writeFileSync(join(outDir, 'encoder.f32'), raw(enc3.data));
  writeFileSync(join(outDir, 'tokens.txt'), out.map(([id, f]) => `${id} ${f}`).join('\n') + '\n');
  writeFileSync(join(outDir, 'text.txt'), text + '\n');
  console.log(`dumped   : ${outDir} (features ${feats.features.dims.join('x')}, encoder ${enc3.dims.join('x')})`);
}
