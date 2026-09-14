// Build a ONE-NODE ONNX model and run it through onnxruntime - a per-operator oracle.
//
// 🔴 WHY. Our engine's own unit tests compare a kernel against a CPU reference WE wrote, and the two can
// share a misreading of the spec - a transposed weight layout is invisible to a test that assumes the same
// transposition. Running the identical operator through onnxruntime settles it against the spec itself.
// It also gives a way to bisect a large model: dump our value for a node, feed the same inputs here, diff.
//
// Writes the model, the inputs and ORT's output next to each other so the C# side can load them.
//
//   node tools/op-oracle.mjs <spec.json> <outdir>
//
// spec.json: { "op": "ConvTranspose", "attrs": {...}, "inputs": [{name,dims,data|randomSeed}], "output": "y" }
// onnx-proto is CommonJS, so it has no named exports under ESM.
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { writeFileSync, readFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-node';

const [specPath, outDir] = process.argv.slice(2);
const spec = JSON.parse(readFileSync(specPath, 'utf8'));
mkdirSync(outDir, { recursive: true });

const FLOAT = onnx.TensorProto.DataType.FLOAT;

// Deterministic pseudo-random, so the C# side can regenerate the identical inputs without shipping them.
function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

const initializers = [], graphInputs = [], feeds = {};
for (const inp of spec.inputs) {
  // An empty name is ONNX's way of saying "this optional input is not supplied" - Resize's `roi` is the
  // common case. It must still occupy its POSITION in the node's input list, or `scales` would be read as
  // `roi` and the operator would silently do something else.
  if (!inp.name) continue;
  const count = inp.dims.reduce((a, b) => a * b, 1);

  // int64 scalars and index tensors - STFT's frame_step and frame_length, Slice's starts/ends. These are
  // always constants in practice, so they go in as initializers and never need feeding.
  if (inp.dtype === 'int64') {
    const big = BigInt64Array.from((inp.data ?? []).map(v => BigInt(v)));
    initializers.push(onnx.TensorProto.create({
      name: inp.name, dataType: onnx.TensorProto.DataType.INT64, dims: inp.dims,
      rawData: new Uint8Array(big.buffer.slice(0)),
    }));
    continue;
  }

  let data;
  if (inp.data) data = Float32Array.from(inp.data);
  else { const r = rng(inp.randomSeed ?? 1); data = Float32Array.from({ length: count }, () => (r() * 2 - 1) * (inp.scale ?? 1)); }
  writeFileSync(join(outDir, `${inp.name}.f32`), Buffer.from(data.buffer));

  if (inp.asInitializer) {
    initializers.push(onnx.TensorProto.create({
      name: inp.name, dataType: FLOAT, dims: inp.dims,
      rawData: new Uint8Array(data.buffer.slice(0)),
    }));
  } else {
    graphInputs.push(onnx.ValueInfoProto.create({
      name: inp.name,
      type: { tensorType: { elemType: FLOAT, shape: { dim: inp.dims.map(d => ({ dimValue: d })) } } },
    }));
    feeds[inp.name] = new ort.Tensor('float32', data, inp.dims);
  }
}

const attributes = Object.entries(spec.attrs ?? {}).map(([key, v]) => {
  if (typeof v === 'string')
    return onnx.AttributeProto.create({ name: key, type: onnx.AttributeProto.AttributeType.STRING, s: Buffer.from(v) });
  if (Array.isArray(v))
    return onnx.AttributeProto.create({ name: key, type: onnx.AttributeProto.AttributeType.INTS, ints: v });
  return onnx.AttributeProto.create({ name: key, type: onnx.AttributeProto.AttributeType.INT, i: v });
});

const model = onnx.ModelProto.create({
  irVersion: 8,
  // ⚠️ Built through the generated type, not as a plain object: a plain {domain,version} silently
  // encodes as opset 0 and ORT then reports "No Op registered ... domain_version of 0", which reads like
  // the operator is unsupported rather than like the model is malformed.
  opsetImport: [onnx.OperatorSetIdProto.create({ domain: '', version: spec.opset ?? 17 })],
  producerName: 'spawndev-op-oracle',
  graph: {
    name: 'g',
    node: [onnx.NodeProto.create({
      opType: spec.op, input: spec.inputs.map(i => i.name), output: [spec.output ?? 'y'], attribute: attributes,
    })],
    input: graphInputs,
    initializer: initializers,
    // Leaving the output shape unspecified lets ORT infer it - which is also a check on OUR shape inference.
    output: [onnx.ValueInfoProto.create({ name: spec.output ?? 'y', type: { tensorType: { elemType: FLOAT } } })],
  },
});

const bytes = onnx.ModelProto.encode(model).finish();
const modelPath = join(outDir, 'model.onnx');
writeFileSync(modelPath, bytes);

const session = await ort.InferenceSession.create(modelPath);
const results = await session.run(feeds);
const out = results[spec.output ?? 'y'];
const data = Float32Array.from(out.data);
writeFileSync(join(outDir, 'expected.f32'), Buffer.from(data.buffer));
// A manifest so the C# side can feed the identical inputs without re-reading the spec: which tensors are
// graph inputs (as opposed to initializers baked into the model), their shapes, and the output name.
writeFileSync(join(outDir, 'meta.json'), JSON.stringify({
  op: spec.op,
  output: spec.output ?? 'y',
  outputDims: out.dims,
  feeds: spec.inputs.filter(i => i.name && !i.asInitializer && i.dtype !== 'int64').map(i => ({ name: i.name, dims: i.dims })),
}, null, 2));

console.log(`OP-ORACLE ${spec.op} -> dims [${out.dims.join(',')}], ${data.length} values`);
console.log(`OP-ORACLE first 8: ${Array.from(data.slice(0, 8)).map(v => v.toFixed(5)).join(' ')}`);
