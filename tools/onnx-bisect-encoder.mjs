// Cut the cache-aware streaming encoder at named tensors, run each cut through onnxruntime on CHUNK 0 (zero caches), and save the cut model + ORT's value: the other half
// of tools/streaming-diff's `--cuts`, which runs the SAME cut through our engine and prints the first tensor that disagrees.
//   node tools/onnx-bisect-encoder.mjs <encoder.onnx> <chunkdump dir> <outDir> <tensor1,tensor2,...>
// The weights live in encoder.data beside the graph; each cut gets a hard link to it so ORT finds them without a 2.4 GB copy.
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import * as ort from 'onnxruntime-node';
import { readFileSync, writeFileSync, mkdirSync, linkSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';

const [modelPath, dumpDir, outDir, namesCsv] = process.argv.slice(2);
if (!namesCsv) { console.error('usage: node tools/onnx-bisect-encoder.mjs <encoder.onnx> <chunkdump dir> <outDir> <names,...>'); process.exit(2); }
mkdirSync(outDir, { recursive: true });
const dataLink = join(outDir, 'encoder.data');
if (!existsSync(dataLink)) linkSync(join(dirname(modelPath), 'encoder.data'), dataLink);

const chunks = readFileSync(join(dumpDir, 'chunks.f32'));
const chunk0 = new Float32Array(chunks.buffer, chunks.byteOffset, 128 * 25);
const feeds = () => ({
  audio_signal: new ort.Tensor('float32', Float32Array.from(chunk0), [1, 128, 25]),
  length: new ort.Tensor('int64', BigInt64Array.from([25n]), [1]),
  cache_last_channel: new ort.Tensor('float32', new Float32Array(24 * 70 * 1024), [1, 24, 70, 1024]),
  cache_last_time: new ort.Tensor('float32', new Float32Array(24 * 1024 * 8), [1, 24, 1024, 8]),
  cache_last_channel_len: new ort.Tensor('int64', BigInt64Array.from([0n]), [1]),
});

const base = onnx.ModelProto.decode(readFileSync(modelPath));
const names = namesCsv.split(',');
const index = [];
for (let i = 0; i < names.length; i++) {
  const [name, elem] = names[i].split('@'); const elemType = elem ? Number(elem) : 1;     // name@9 = a bool tensor, name@7 = int64: saved as float32 0/1 and integers, as the engine holds them
  const m = onnx.ModelProto.decode(readFileSync(modelPath));
  m.graph.output = [onnx.ValueInfoProto.create({ name, type: { tensorType: { elemType } } })];
  const file = join(outDir, `cut_${i}.onnx`);
  writeFileSync(file, onnx.ModelProto.encode(m).finish());
  try {
    const s = await ort.InferenceSession.create(file, { graphOptimizationLevel: 'disabled', executionProviders: ['cpu'] });
    const r = await s.run(feeds());
    const t = r[name];
    const asFloat = Float32Array.from(t.data, v => Number(v));
    writeFileSync(join(outDir, `cut_${i}.f32`), Buffer.from(asFloat.buffer));
    index.push({ i, name, dims: t.dims.map(Number) });
    console.log(i, name, '[' + t.dims.join(',') + ']');
  } catch (e) { console.log(i, name, 'ORT could not run this cut:', String(e.message).slice(0, 120)); }
}
writeFileSync(join(outDir, 'index.json'), JSON.stringify(index));
