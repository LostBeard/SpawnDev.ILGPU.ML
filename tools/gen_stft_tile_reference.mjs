// Fixtures for two operator bugs found bringing up Parakeet TDT (its log-mel graph and its FastConformer encoder).
//
//   node tools/gen_stft_tile_reference.mjs
//
// 1. stft_long_window: STFT with a 512-sample Hann window, hop 160, on a rank-2 signal [1, N] - the shape of NeMo's
//    preprocessor. The engine only read initializers of <= 64 elements as host values, so the 512-sample window was
//    silently DROPPED and frame_length fell back to the hop (160): right shape, wrong values (97% error).
//    Also pins the frame count: the hop is an input VALUE and used to be folded after shape inference, so 64,000
//    samples were inferred as 501 frames (hop 128) instead of 401.
// 2. tile_computed: Tile of a COMPUTED tensor, [1,1,T] repeated [1,T,1] - the relative-position index of a conformer.
//    The general N-D Tile needed the input's values on the host, which only exist for small constants, so it threw
//    "input values are not available on the CPU".
//
// Expected values come from onnxruntime (onnxruntime-node), optimizations off. Same JSON shape as the other
// references/fusion fixtures.
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import * as ort from 'onnxruntime-node';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'SpawnDev.ILGPU.ML.Demo', 'wwwroot', 'references', 'fusion');
mkdirSync(OUT, { recursive: true });
const DT = onnx.TensorProto.DataType;

const vi = (name, type, dims) => onnx.ValueInfoProto.create({ name, type: { tensorType: { elemType: type, shape: { dim: dims.map(d => ({ dimValue: d })) } } } });
const node = (opType, inputs, outputs, attrs = []) => onnx.NodeProto.create({ opType, input: inputs, output: outputs, attribute: attrs });
const tensorF = (name, dims, data) => onnx.TensorProto.create({ name, dims, dataType: DT.FLOAT, floatData: Array.from(data) });
const tensorI64 = (name, dims, data) => {
  const raw = Buffer.alloc(8 * data.length); data.forEach((v, i) => raw.writeBigInt64LE(BigInt(v), i * 8));
  return onnx.TensorProto.create({ name, dims, dataType: DT.INT64, rawData: raw });
};
const model = (name, nodes, inits, inputs, outputs) => onnx.ModelProto.create({
  irVersion: 8, producerName: 'pytorch', opsetImport: [{ domain: '', version: 17 }],
  graph: { name, node: nodes, initializer: inits, input: inputs, output: outputs },
});

async function save(name, m, feeds, outputNames) {
  const bytes = onnx.ModelProto.encode(m).finish();
  writeFileSync(join(OUT, `${name}.onnx`), bytes);
  const session = await ort.InferenceSession.create(bytes, { graphOptimizationLevel: 'disabled', executionProviders: ['cpu'] });
  const res = await session.run(feeds);
  const ref = {
    inputs: Object.fromEntries(Object.entries(feeds).map(([k, t]) => [k, { shape: t.dims.map(Number), data: Array.from(t.data, Number) }])),
    outputs: Object.fromEntries(outputNames.map(n => [n, { shape: res[n].dims.map(Number), data: Array.from(res[n].data, Number) }])),
  };
  writeFileSync(join(OUT, `${name}.json`), JSON.stringify(ref));
  console.log(`${name} -> ${bytes.length} bytes; ${outputNames.map(n => `${n} [${res[n].dims.join(',')}]`).join(', ')}`);
}

// ── 1. STFT, 512-sample Hann window, hop 160, rank-2 signal ──
{
  const N = 4160;                                   // (4160 - 512) / 160 + 1 = 23 frames (a count that is not 128-hop-shaped)
  let seed = 7;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return ((seed / 4294967296) - 0.5) * 2; };
  const signal = Float32Array.from({ length: N }, (_, i) => 0.6 * rnd() + 0.3 * Math.sin(i * 0.05));
  const hann = Float32Array.from({ length: 512 }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / 512));
  const m = model('stft_long_window',
    [node('STFT', ['x', 'hop', 'win'], ['y'])],
    [tensorI64('hop', [], [160]), tensorF('win', [512], hann)],
    [vi('x', DT.FLOAT, [1, N])], [vi('y', DT.FLOAT, [1, 23, 257, 2])]);
  await save('stft_long_window', m, { x: new ort.Tensor('float32', signal, [1, N]) }, ['y']);
}

// ── 2. Tile of a computed [1,1,T] by [1,T,1] ──
{
  const T = 9;
  const m = model('tile_computed',
    [
      node('Shape', ['x'], ['xs']),                                                           // [1, T]
      node('Gather', ['xs', 'one'], ['t']),                                                   // T (scalar)
      node('Range', ['zero', 't', 'one'], ['r']),                                             // [0..T)
      node('Unsqueeze', ['r', 'axes01'], ['r3']),                                             // [1,1,T]
      node('Unsqueeze', ['t', 'axes0'], ['t1']),                                              // [T]
      node('Concat', ['one1', 't1', 'one1'], ['reps'], [onnx.AttributeProto.create({ name: 'axis', type: onnx.AttributeProto.AttributeType.INT, i: 0 })]),
      node('Tile', ['r3', 'reps'], ['tiled']),                                                // [1,T,T]
      node('Cast', ['tiled'], ['y'], [onnx.AttributeProto.create({ name: 'to', type: onnx.AttributeProto.AttributeType.INT, i: DT.FLOAT })]),
    ],
    [tensorI64('one', [], [1]), tensorI64('zero', [], [0]), tensorI64('axes01', [2], [0, 1]), tensorI64('axes0', [1], [0]), tensorI64('one1', [1], [1])],
    [vi('x', DT.FLOAT, [1, T])], [vi('y', DT.FLOAT, [1, T, T])]);
  await save('tile_computed', m, { x: new ort.Tensor('float32', new Float32Array(T), [1, T]) }, ['y']);
}
