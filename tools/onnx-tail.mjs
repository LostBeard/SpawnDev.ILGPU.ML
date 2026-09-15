// Walk BACKWARDS from a graph output to the first node of a given op type, and print the whole tail in
// topological order with every initializer it reads.
//
// 🔴 WHY. Truncating a graph and reproducing its tail on the host is only safe if you can SEE the tail.
// "It divides by a window then scales by 4" is a memory of a debug session; this prints the nodes, so the
// host implementation is derived from the model and a different export is detected instead of mis-run.
//
//   node tools/onnx-tail.mjs <model.onnx> <stop-op-type> [output-name]
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { readFileSync } from 'node:fs';

const [modelPath, stopOp, outName] = process.argv.slice(2);
const g = onnx.ModelProto.decode(readFileSync(modelPath)).graph;
const producer = new Map();
for (const n of g.node) for (const o of n.output) producer.set(o, n);
const inits = new Map(g.initializer.map(t => [t.name, t]));

function initVals(t) {
  const count = t.dims.reduce((a, b) => a * Number(b), 1);
  const buf = Buffer.from(t.rawData ?? []);
  const D = onnx.TensorProto.DataType;
  let vals = [];
  if (t.floatData?.length) vals = Array.from(t.floatData);
  else if (t.int64Data?.length) vals = t.int64Data.map(Number);
  else if (buf.length && t.dataType === D.FLOAT) { for (let i = 0; i < count; i++) vals.push(buf.readFloatLE(i * 4)); }
  else if (buf.length && t.dataType === D.INT64) { for (let i = 0; i < count; i++) vals.push(Number(buf.readBigInt64LE(i * 8))); }
  else if (buf.length && t.dataType === D.INT32) { for (let i = 0; i < count; i++) vals.push(buf.readInt32LE(i * 4)); }
  return vals;
}

const target = outName ?? g.output[0].name;
// Reverse BFS, stopping the descent at any node of stopOp (that node is the new graph output).
const collected = new Map();      // node name -> node
const boundary = new Set();       // tensor names the tail consumes from ABOVE the cut
const queue = [target];
const seen = new Set();
while (queue.length) {
  const t = queue.shift();
  if (seen.has(t)) continue;
  seen.add(t);
  const n = producer.get(t);
  if (!n) { boundary.add(t); continue; }
  if (n.opType === stopOp) { boundary.add(t); continue; }
  collected.set(n.name || n.output[0], n);
  for (const i of n.input) if (i) queue.push(i);
}

// Topological order: a node is ready when every input is an initializer, a boundary tensor, or already emitted.
const ready = new Set([...boundary, ...inits.keys()]);
const order = [];
const pending = [...collected.values()];
while (pending.length) {
  const i = pending.findIndex(n => n.input.every(x => !x || ready.has(x)));
  if (i < 0) { console.log('!! cycle or unresolved inputs in', pending.map(n => n.opType).join(',')); break; }
  const [n] = pending.splice(i, 1);
  order.push(n);
  for (const o of n.output) ready.add(o);
}

console.log(`tail of ${modelPath} from output "${target}" back to the nearest ${stopOp}`);
console.log(`cut boundary (new graph outputs): ${[...boundary].join(', ')}`);
console.log(`${order.length} node(s) downstream of the cut\n`);
for (const n of order) {
  const attrs = n.attribute.map(a => {
    if (a.ints?.length) return `${a.name}=[${a.ints.map(Number).join(',')}]`;
    if (a.floats?.length) return `${a.name}=[${a.floats.join(',')}]`;
    if (a.s?.length) return `${a.name}="${Buffer.from(a.s).toString()}"`;
    if (a.i !== undefined && a.i !== null) return `${a.name}=${Number(a.i)}`;
    if (a.f !== undefined && a.f !== null) return `${a.name}=${a.f}`;
    if (a.t) return `${a.name}=<tensor ${a.t.dims.join('x')} ${initVals(a.t).slice(0, 8).join(',')}>`;
    return a.name;
  });
  console.log(`${n.opType}  ${n.name}`);
  console.log(`   in : ${n.input.join(' , ')}`);
  console.log(`   out: ${n.output.join(' , ')}`);
  if (attrs.length) console.log(`   att: ${attrs.join('  ')}`);
  for (const i of n.input) {
    const t = inits.get(i);
    if (!t) continue;
    const v = initVals(t);
    const count = t.dims.reduce((a, b) => a * Number(b), 1);
    const show = v.length > 24 ? v.slice(0, 24).join(',') + ` ... (${count} total)` : v.join(',');
    console.log(`   INIT ${i} dims=[${t.dims.map(Number).join(',')}] : ${show}`);
  }
  console.log('');
}
