// Print the nodes whose name contains a substring, in graph order: op, name suffix, inputs (initializers marked),
// outputs. For reading a block (an attention, a norm) the way the engine will see it.
//   node tools/onnx-nodes-between.mjs <model.onnx> <name substring>
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { readFileSync } from 'node:fs';
const [path, sub] = process.argv.slice(2);
const g = onnx.ModelProto.decode(readFileSync(path)).graph;
const inits = new Set(g.initializer.map(i => i.name));
const consts = new Map();
for (const n of g.node) if (n.opType === 'Constant') consts.set(n.output[0], n);
const short = s => s.split('/').slice(-2).join('/');
for (const n of g.node) {
  if (!(n.name || '').includes(sub)) continue;
  const attrs = n.attribute.map(a => `${a.name}=${a.i !== undefined && a.type === 2 ? a.i : a.type === 1 ? a.f : a.ints?.length ? '[' + a.ints.join(',') + ']' : a.s ? Buffer.from(a.s).toString() : ''}`).join(' ');
  const ins = n.input.map(i => inits.has(i) ? `<${short(i)}>` : consts.has(i) ? '<const>' : short(i)).join(', ');
  console.log(`${n.opType.padEnd(10)} ${short(n.name).padEnd(28)} (${ins}) -> ${short(n.output[0])} ${attrs}`);
}
