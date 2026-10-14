// Fixture for the binary element-wise operators: two operands with the SAME element count but DIFFERENT shapes broadcast to a
// larger output ([N,1] op [1,N] -> [N,N], the outer product shape).
//
//   node tools/gen_outer_broadcast_reference.mjs
//
// WHY: SubOperator took `a.ElementCount == b.ElementCount` as "same shape" and subtracted element by element over
// min(count, outputCount) values. The Nemotron streaming encoder builds its attention mask from Sub([72,1], [1,72]) (a position
// difference): indices 72.. were never written and the first 72 held a[i]-b[i] - the mask came out wrong, softmax attended where it
// must not, and WER went 5.6% -> 7.2% with no error anywhere. The other operators are gated here too, so they cannot regress.
// Expected values come from onnxruntime (onnxruntime-node), graph optimizations off. Same JSON shape as the other references/fusion fixtures.
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import * as ort from 'onnxruntime-node';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'SpawnDev.ILGPU.ML.Demo', 'wwwroot', 'references', 'fusion');
mkdirSync(OUT, { recursive: true });

// Deterministic, asymmetric values: a[i] != b[i] for every i, so an element-wise result can never equal the outer one.
let seed = 20261014;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const N = 8;
const a = Float32Array.from({ length: N }, () => 1 + rnd() * 4);       // 1..5
const b = Float32Array.from({ length: N }, () => 0.5 + rnd() * 3);     // 0.5..3.5 (non-zero: Div)

const vi = (name, dims) => onnx.ValueInfoProto.create({
  name, type: { tensorType: { elemType: onnx.TensorProto.DataType.FLOAT, shape: { dim: dims.map(d => ({ dimValue: d })) } } },
});
const ops = ['Add', 'Sub', 'Mul', 'Div', 'Min', 'Max'];
const model = onnx.ModelProto.create({
  irVersion: 8,
  producerName: 'pytorch',
  opsetImport: [{ domain: '', version: 17 }],
  graph: {
    name: 'outer_broadcast',
    node: ops.map(op => onnx.NodeProto.create({ opType: op, input: ['A', 'B'], output: [op + '_out'] })),
    input: [vi('A', [N, 1]), vi('B', [1, N])],
    output: ops.map(op => vi(op + '_out', [N, N])),
  },
});
const bytes = onnx.ModelProto.encode(model).finish();
writeFileSync(join(OUT, 'outer_broadcast.onnx'), bytes);

const session = await ort.InferenceSession.create(bytes, { graphOptimizationLevel: 'disabled', executionProviders: ['cpu'] });
const res = await session.run({ A: new ort.Tensor('float32', a, [N, 1]), B: new ort.Tensor('float32', b, [1, N]) });
const ref = {
  inputs: { A: { shape: [N, 1], data: Array.from(a) }, B: { shape: [1, N], data: Array.from(b) } },
  outputs: Object.fromEntries(ops.map(op => [op + '_out', { shape: res[op + '_out'].dims.map(Number), data: Array.from(res[op + '_out'].data) }])),
};
writeFileSync(join(OUT, 'outer_broadcast.json'), JSON.stringify(ref));
console.log(`outer_broadcast -> ${bytes.length} bytes, ${ops.join(', ')} each ${N}x${N}`);
