// ReDimNet2 speaker embeddings on onnxruntime-node: the REFERENCE for SpeakerEmbeddingPipeline.
//
//   node tools/speaker-oracle.mjs <ReDimNet2B6.onnx> <corpusDir of 16 kHz mono wavs> <out.json> [maxSeconds]
//
// Every WAV is fitted to the model's fixed 6 s window the way the pipeline does it (shorter: repeated to fill,
// longer: centre-cropped; with maxSeconds, the clip is first cut to its first maxSeconds), embedded, and written as
// {id: [192 floats]}. tools/speaker-bench compares our embeddings with these (cosine, per clip).
import * as ort from 'onnxruntime-node';
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const [modelPath, corpus, outPath, maxSecondsArg] = process.argv.slice(2);
if (!outPath) { console.error('usage: node tools/speaker-oracle.mjs <model.onnx> <corpusDir> <out.json> [maxSeconds]'); process.exit(2); }
const WINDOW = 96000;
const maxSamples = maxSecondsArg ? Math.round(Number(maxSecondsArg) * 16000) : Infinity;

function wav(path) {
  const b = readFileSync(path);
  let p = 12, data = null;
  while (p + 8 <= b.length) {
    const id = b.toString('ascii', p, p + 4), size = b.readUInt32LE(p + 4);
    if (id === 'data') { data = b.subarray(p + 8, Math.min(b.length, p + 8 + size)); break; }
    p += 8 + size + (size & 1);
  }
  const out = new Float32Array(data.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = data.readInt16LE(i * 2) / 32768;
  return out;
}

const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
const result = {};
const files = readdirSync(corpus).filter(f => f.endsWith('.wav')).sort();
const t0 = performance.now();
for (const f of files) {
  let pcm = wav(join(corpus, f));
  if (pcm.length > maxSamples) pcm = pcm.subarray(0, maxSamples);
  const window = new Float32Array(WINDOW);
  if (pcm.length >= WINDOW) { const off = Math.floor((pcm.length - WINDOW) / 2); window.set(pcm.subarray(off, off + WINDOW)); }
  else for (let i = 0; i < WINDOW; i++) window[i] = pcm[i % pcm.length];
  const out = await session.run({ audio: new ort.Tensor('float32', window, [1, WINDOW]) });
  result[f.replace(/\.wav$/, '')] = Array.from(out.embedding.data);
}
writeFileSync(outPath, JSON.stringify(result));
console.log(`${files.length} embeddings -> ${outPath} in ${((performance.now() - t0) / 1000).toFixed(1)} s (CPU, onnxruntime-node)`);
