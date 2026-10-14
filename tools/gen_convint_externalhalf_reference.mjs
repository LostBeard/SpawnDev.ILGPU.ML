// Fixtures for two defects found loading Parakeet TDT's int8 and fp16 encoders.
//
//   node tools/gen_convint_externalhalf_reference.mjs
//
// 1. convinteger: ConvInteger as the quantized FastConformer uses it - NON-square stride, asymmetric pads, dilation, a depthwise
//    group, and a rank-3 (1-D) convolution; scalar w_zero_point (onnxruntime rejects a per-channel one; the engine implements it per the ONNX spec but nothing here can confirm it). The old implementation read the activation
//    on the HOST (only possible for constants of <= 64 elements), fell back to Conv2D with an EMPTY bias view (CUDA illegal memory
//    access) and ignored the zero points; rank 3 returned zeros.
// 2. external_half: a model whose weights live in a .onnx_data file (data_location = EXTERNAL), loaded with WeightStorage.Half.
//    One weight is the weight operand of a MatMul (eligible: must be stored fp16), one is used by an elementwise Mul (must stay
//    fp32), one is a rank-3 Conv weight (must stay fp32: the half Conv path is 2-D only). The external-data loop used to ignore
//    WeightStorage.Half entirely, and the half-eligibility rule admitted rank-3 convs.
//
// Expected values come from onnxruntime (onnxruntime-node) on the files as written, optimizations off.
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import * as ort from 'onnxruntime-node';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'SpawnDev.ILGPU.ML.Demo', 'wwwroot', 'references', 'fusion');
mkdirSync(OUT, { recursive: true });
const DT = onnx.TensorProto.DataType;
const ATTR = onnx.AttributeProto.AttributeType;

let seed = 20261014;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const ints = (n, lo, hi) => Float32Array.from({ length: n }, () => lo + Math.floor(rnd() * (hi - lo + 1)));
const floats = n => Float32Array.from({ length: n }, () => (rnd() - 0.5) * 2);

// dims null: no declared shape (an empty list would DECLARE a scalar, which the engine pre-registers as the output's shape)
const vi = (name, type, dims) => onnx.ValueInfoProto.create({ name, type: { tensorType: dims ? { elemType: type, shape: { dim: dims.map(d => ({ dimValue: d })) } } : { elemType: type } } });
const attr = (name, v) => Array.isArray(v)
  ? onnx.AttributeProto.create({ name, type: ATTR.INTS, ints: v })
  : onnx.AttributeProto.create({ name, type: ATTR.INT, i: v });
const node = (opType, inputs, outputs, attrs = {}) => onnx.NodeProto.create({ opType, input: inputs, output: outputs, attribute: Object.entries(attrs).map(([k, v]) => attr(k, v)) });
const u8 = (name, dims, data) => onnx.TensorProto.create({ name, dims, dataType: DT.UINT8, rawData: Buffer.from(Uint8Array.from(data)) });
const f32 = (name, dims, data) => onnx.TensorProto.create({ name, dims, dataType: DT.FLOAT, floatData: Array.from(data) });
const model = (name, nodes, inits, inputs, outputs) => onnx.ModelProto.create({
  irVersion: 8, producerName: 'pytorch', opsetImport: [{ domain: '', version: 17 }],
  graph: { name, node: nodes, initializer: inits, input: inputs, output: outputs },
});
const toJson = (feeds, res, names) => ({
  inputs: Object.fromEntries(Object.entries(feeds).map(([k, t]) => [k, { shape: t.dims.map(Number), data: Array.from(t.data, Number) }])),
  outputs: Object.fromEntries(names.map(n => [n, { shape: res[n].dims.map(Number), data: Array.from(res[n].data, Number) }])),
});

// ── 1. ConvInteger ──
{
  const xs = [1, 3, 9, 11], x2 = [1, 4, 10, 10], x3 = [1, 2, 16];
  const castU8 = (i, o) => node('Cast', [i], [o], { to: DT.UINT8 });
  const castF = (i, o) => node('Cast', [i], [o], { to: DT.FLOAT });
  const m = model('convinteger',
    [
      castU8('xa', 'xa8'), node('ConvInteger', ['xa8', 'wa', 'zxa', 'zwa'], ['ya_i'], { strides: [2, 1], pads: [1, 0, 1, 2], dilations: [1, 2] }), castF('ya_i', 'ya'),
      castU8('xb', 'xb8'), node('ConvInteger', ['xb8', 'wb', 'zxb', 'zwb'], ['yb_i'], { group: 4, pads: [1, 1, 1, 1] }), castF('yb_i', 'yb'),
      castU8('xc', 'xc8'), node('ConvInteger', ['xc8', 'wc', 'zxc', 'zwc'], ['yc_i'], { strides: [2], pads: [1, 1] }), castF('yc_i', 'yc'),
    ],
    [
      u8('wa', [5, 3, 3, 3], ints(5 * 3 * 3 * 3, 0, 255)), u8('zxa', [], [113]), u8('zwa', [], [87]),
      u8('wb', [4, 1, 3, 3], ints(4 * 9, 0, 255)), u8('zxb', [], [128]), u8('zwb', [], [120]),                     // a depthwise convolution (onnxruntime supports only a per-tensor w_zero_point, so that is what is referenced)
      u8('wc', [3, 2, 3], ints(3 * 2 * 3, 0, 255)), u8('zxc', [], [7]), u8('zwc', [], [201]),                       // a 1-D convolution
    ],
    [vi('xa', DT.FLOAT, xs), vi('xb', DT.FLOAT, x2), vi('xc', DT.FLOAT, x3)],
    [vi('ya', DT.FLOAT, null), vi('yb', DT.FLOAT, null), vi('yc', DT.FLOAT, null)]);
  const bytes = onnx.ModelProto.encode(m).finish();
  writeFileSync(join(OUT, 'convinteger.onnx'), bytes);
  const feeds = {
    xa: new ort.Tensor('float32', ints(xs.reduce((a, b) => a * b), 0, 255), xs),
    xb: new ort.Tensor('float32', ints(x2.reduce((a, b) => a * b), 0, 255), x2),
    xc: new ort.Tensor('float32', ints(x3.reduce((a, b) => a * b), 0, 255), x3),
  };
  const session = await ort.InferenceSession.create(bytes, { graphOptimizationLevel: 'disabled', executionProviders: ['cpu'] });
  const res = await session.run(feeds);
  writeFileSync(join(OUT, 'convinteger.json'), JSON.stringify(toJson(feeds, res, ['ya', 'yb', 'yc'])));
  console.log('convinteger ->', ['ya', 'yb', 'yc'].map(n => `${n} [${res[n].dims.join(',')}]`).join(', '));
}

// ── 2. external-data weights + WeightStorage.Half ──
{
  const W = floats(256 * 128), B = floats(128), S = floats(256), Wc = floats(4 * 2 * 3);
  const dataFile = 'external_half.onnx.data';
  const chunks = []; let offset = 0;
  const external = (name, dims, data) => {
    const raw = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
    const entry = onnx.TensorProto.create({
      name, dims, dataType: DT.FLOAT, dataLocation: onnx.TensorProto.DataLocation.EXTERNAL,
      externalData: [{ key: 'location', value: dataFile }, { key: 'offset', value: String(offset) }, { key: 'length', value: String(raw.length) }],
    });
    chunks.push(raw); offset += raw.length;
    return entry;
  };
  const m = model('external_half',
    [
      node('MatMul', ['x', 'W'], ['mm']), node('Add', ['mm', 'B'], ['y1']),
      node('Mul', ['x', 'S'], ['y2']),
      node('Conv', ['xc', 'Wc'], ['y3'], { group: 1 }),
    ],
    [external('W', [256, 128], W), f32('B', [128], B), external('S', [256], S), external('Wc', [4, 2, 3], Wc)],
    [vi('x', DT.FLOAT, [1, 8, 256]), vi('xc', DT.FLOAT, [1, 2, 16])],
    [vi('y1', DT.FLOAT, [1, 8, 128]), vi('y2', DT.FLOAT, [1, 8, 256]), vi('y3', DT.FLOAT, [1, 4, 14])]);
  writeFileSync(join(OUT, 'external_half.onnx'), onnx.ModelProto.encode(m).finish());
  writeFileSync(join(OUT, dataFile), Buffer.concat(chunks));
  const feeds = { x: new ort.Tensor('float32', floats(8 * 256), [1, 8, 256]), xc: new ort.Tensor('float32', floats(2 * 16), [1, 2, 16]) };
  const session = await ort.InferenceSession.create(join(OUT, 'external_half.onnx'), { graphOptimizationLevel: 'disabled', executionProviders: ['cpu'] });
  const res = await session.run(feeds);
  writeFileSync(join(OUT, 'external_half.json'), JSON.stringify(toJson(feeds, res, ['y1', 'y2', 'y3'])));
  console.log('external_half ->', ['y1', 'y2', 'y3'].map(n => `${n} [${res[n].dims.join(',')}]`).join(', '), `data ${offset} bytes`);
}
