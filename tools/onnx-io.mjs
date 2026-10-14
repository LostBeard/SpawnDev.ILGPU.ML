// Print an ONNX model's graph inputs/outputs (name, dtype, shape) - the contract a pipeline must honour.
//   node tools/onnx-io.mjs <model.onnx> [more.onnx ...]
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { readFileSync } from 'node:fs';

const dims = t => (t.tensorType?.shape?.dim ?? []).map(d => d.dimParam || d.dimValue?.toString() || '?').join(', ');
for (const path of process.argv.slice(2)) {
  const g = onnx.ModelProto.decode(readFileSync(path)).graph;
  const init = new Set(g.initializer.map(i => i.name));
  console.log(`== ${path}`);
  for (const i of g.input.filter(i => !init.has(i.name)))
    console.log(`  in  ${i.name.padEnd(24)} type ${i.type.tensorType?.elemType} [${dims(i.type)}]`);
  for (const o of g.output)
    console.log(`  out ${o.name.padEnd(24)} type ${o.type.tensorType?.elemType} [${dims(o.type)}]`);
}
