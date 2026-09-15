// Count op types per NAME PREFIX in an ONNX graph.
//
// 🔴 WHY. In a browser this engine pays per DISPATCH, not per FLOP - MEASURED on Kokoro/CUDA, the
// encoder/bert half is 602 of 1,850 nodes and 1.6% of the compute. So the question "where is the browser
// time?" is answered by a NODE CENSUS, not by a profiler: whichever subtree has the most nodes costs the
// most, however trivial its arithmetic. This prints that census so fusion work aims at the right subtree.
//
//   node tools/onnx-opcensus.mjs <model.onnx> [prefix]
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { readFileSync } from 'node:fs';

const [modelPath, prefix] = process.argv.slice(2);
const g = onnx.ModelProto.decode(readFileSync(modelPath)).graph;
const nodes = prefix ? g.node.filter(n => (n.name || '').includes(prefix)) : g.node;

const byOp = new Map();
for (const n of nodes) byOp.set(n.opType, (byOp.get(n.opType) ?? 0) + 1);
console.log(`${nodes.length} nodes${prefix ? ` under "${prefix}"` : ''} of ${g.node.length} total\n`);
for (const [op, c] of [...byOp].sort((a, b) => b[1] - a[1]))
  console.log(`  ${String(c).padStart(5)}  ${op}`);

// Top-level subtrees, so "which half" is answerable without knowing the naming scheme in advance.
if (!prefix) {
  const bySub = new Map();
  for (const n of g.node) {
    const parts = (n.name || '').split('/').filter(Boolean);
    const key = parts.slice(0, 3).join('/') || '(unnamed)';
    bySub.set(key, (bySub.get(key) ?? 0) + 1);
  }
  console.log('\nby subtree (first 3 name segments):');
  for (const [k, c] of [...bySub].sort((a, b) => b[1] - a[1]).slice(0, 25))
    console.log(`  ${String(c).padStart(5)}  ${k}`);
}
