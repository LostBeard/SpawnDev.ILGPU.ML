// Print the nodes of an ONNX graph whose name matches a substring, with their inputs, outputs and
// attributes - including which initializers feed them and what those initializers contain.
//
// 🔴 WHY. Diffing intermediates tells you WHICH node diverged; it cannot tell you WHY. The why is almost
// always an attribute we ignored (a Resize coordinate_transformation_mode, a Pad mode, an axis default).
// Reading the attribute off the real model beats assuming the common case.
//
//   node tools/onnx-nodes.mjs <model.onnx> <substring>
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { readFileSync } from 'node:fs';

const [modelPath, filter] = process.argv.slice(2);
const model = onnx.ModelProto.decode(readFileSync(modelPath));
const g = model.graph;

// Initializers are looked up by name so a small one can be printed inline - a Resize's `scales` is the
// difference between "upsamples 300x" and "downsamples", and it is never in the node itself.
const inits = new Map(g.initializer.map(t => [t.name, t]));
function initText(name) {
  const t = inits.get(name);
  if (!t) return null;
  const count = t.dims.reduce((a, b) => a * Number(b), 1);
  if (count > 16) return `<init ${t.dims.join('x')}>`;
  const buf = Buffer.from(t.rawData ?? []);
  let vals = [];
  if (t.floatData?.length) vals = Array.from(t.floatData);
  else if (t.int64Data?.length) vals = t.int64Data.map(Number);
  else if (buf.length && t.dataType === onnx.TensorProto.DataType.FLOAT)
    for (let i = 0; i < count; i++) vals.push(buf.readFloatLE(i * 4));
  else if (buf.length && t.dataType === onnx.TensorProto.DataType.INT64)
    for (let i = 0; i < count; i++) vals.push(Number(buf.readBigInt64LE(i * 8)));
  return `<init ${t.dims.join('x')} = ${vals.map(v => (typeof v === 'number' ? +v.toFixed(5) : v)).join(',')}>`;
}

function attrText(a) {
  const T = onnx.AttributeProto.AttributeType;
  switch (a.type) {
    case T.STRING: return Buffer.from(a.s).toString();
    case T.INT: return String(a.i);
    case T.FLOAT: return String(a.f);
    case T.INTS: return `[${a.ints.join(',')}]`;
    case T.FLOATS: return `[${a.floats.join(',')}]`;
    default: return `<type ${a.type}>`;
  }
}

let n = 0;
for (const node of g.node) {
  const tag = node.name || node.output[0] || '';
  if (filter && !tag.includes(filter)) continue;
  n++;
  const attrs = node.attribute.map(a => `${a.name}=${attrText(a)}`).join(' ');
  console.log(`${node.opType.padEnd(14)} ${tag}`);
  for (const i of node.input) console.log(`    in  ${i} ${initText(i) ?? ''}`);
  for (const o of node.output) console.log(`    out ${o}`);
  if (attrs) console.log(`    attr ${attrs}`);
}
console.log(`${n} nodes matched`);
