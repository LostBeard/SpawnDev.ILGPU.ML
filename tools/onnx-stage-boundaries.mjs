// List a graph's nodes in order with their first output name, collapsing runs by the first N path segments of the node
// name, so the boundaries between big blocks (stages, layers) can be picked as bisect cut points.
//   node tools/onnx-stage-boundaries.mjs <model.onnx> [segments=3]
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { readFileSync } from 'node:fs';
const [path, segArg] = process.argv.slice(2);
const seg = Number(segArg ?? 3);
const g = onnx.ModelProto.decode(readFileSync(path)).graph;
let cur = null, first = null, last = null, count = 0;
const flush = () => cur !== null && console.log(`${String(count).padStart(5)} nodes  ${cur.padEnd(40)} first-out ${first}   last-out ${last}`);
for (const n of g.node) {
  if (n.opType === 'Constant') continue;
  const key = (n.name || '').split('/').filter(Boolean).slice(0, seg).join('/') || '(unnamed)';
  if (key !== cur) { flush(); cur = key; first = n.output[0]; count = 0; }
  last = n.output[0]; count++;
}
flush();
