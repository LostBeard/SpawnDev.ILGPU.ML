// Fixture for GraphOptimizer.FuseLinearLayers: a MatMul followed by an Add of a SCALAR constant is not a bias.
//
//   node tools/gen_matmul_scalar_add_reference.mjs
//
// WHY: Parakeet's log-mel preprocessor (nemo128.onnx) ends in `MatMul([T,257] x [257,128]) -> Add(0 scalar) -> Log`.
// FuseLinearLayers only asked "is the Add's other input an initializer?", so it built a FusedLinear whose "bias" was
// one element for 128 columns: "FusedLinear bounds mismatch: M=501 K=257 N=128 ... bias=[] buf=1".
//
// Two paths in one graph, so the test pins BOTH directions:
//   scalar_out = MatMul(x, W1) + 0.25      (scalar: must stay MatMul + Add)
//   bias_out   = MatMul(x, W2) + b[N]      (a real bias: must still fuse into FusedLinear)
// Expected values come from onnxruntime (onnxruntime-node), graph optimizations off. Same JSON shape as the
// other references/fusion fixtures (tools/gen_fusion_reference.py).
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import * as ort from 'onnxruntime-node';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'SpawnDev.ILGPU.ML.Demo', 'wwwroot', 'references', 'fusion');
mkdirSync(OUT, { recursive: true });

// A deterministic, non-friendly value stream (no zeros, no symmetry): a scalar bias of 0 would hide the bug in
// the value check, and a [K,K] weight would hide a K/N mix-up.
let seed = 20261014;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return ((seed / 4294967296) - 0.5) * 2; };
const fill = n => Float32Array.from({ length: n }, rnd);

const M = 6, K = 9, N = 5;
const x = fill(1 * M * K), w1 = fill(K * N), w2 = fill(K * N), b = fill(N);
const SCALAR = 0.25;

const tensor = (name, dims, data) => onnx.TensorProto.create({ name, dims, dataType: onnx.TensorProto.DataType.FLOAT, floatData: Array.from(data) });
const vi = (name, dims) => onnx.ValueInfoProto.create({
  name, type: { tensorType: { elemType: onnx.TensorProto.DataType.FLOAT, shape: { dim: dims.map(d => ({ dimValue: d })) } } },
});
const node = (opType, inputs, outputs) => onnx.NodeProto.create({ opType, input: inputs, output: outputs });

const model = onnx.ModelProto.create({
  irVersion: 8,
  producerName: 'pytorch',   // an exporter name DetectModelFormat accepts in any case
  opsetImport: [{ domain: '', version: 17 }],
  graph: {
    name: 'matmul_scalar_add',
    node: [
      node('MatMul', ['X', 'W1'], ['mm1']), node('Add', ['mm1', 'scalar'], ['scalar_out']),
      node('MatMul', ['X', 'W2'], ['mm2']), node('Add', ['mm2', 'B'], ['bias_out']),
    ],
    initializer: [tensor('W1', [K, N], w1), tensor('W2', [K, N], w2), tensor('B', [N], b), tensor('scalar', [], [SCALAR])],
    input: [vi('X', [1, M, K])],
    output: [vi('scalar_out', [1, M, N]), vi('bias_out', [1, M, N])],
  },
});
const bytes = onnx.ModelProto.encode(model).finish();
writeFileSync(join(OUT, 'matmul_scalar_add.onnx'), bytes);

const session = await ort.InferenceSession.create(bytes, { graphOptimizationLevel: 'disabled', executionProviders: ['cpu'] });
const res = await session.run({ X: new ort.Tensor('float32', x, [1, M, K]) });
const ref = {
  inputs: { X: { shape: [1, M, K], data: Array.from(x) } },
  outputs: Object.fromEntries(['scalar_out', 'bias_out'].map(n => [n, { shape: res[n].dims.map(Number), data: Array.from(res[n].data) }])),
};
writeFileSync(join(OUT, 'matmul_scalar_add.json'), JSON.stringify(ref));
console.log(`matmul_scalar_add -> ${bytes.length} bytes, outputs ${Object.keys(ref.outputs).join(', ')} (${M}x${N} each), scalar ${SCALAR}`);
