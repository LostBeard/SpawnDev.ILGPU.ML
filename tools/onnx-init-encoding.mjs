// How each SMALL initializer of an ONNX file is stored (dtype, dims, which data field holds it).
// Loaders that read only one field (raw_data vs int64_data vs float_data) silently miss the others.
//   node tools/onnx-init-encoding.mjs <model.onnx> [maxElements=16]
import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { readFileSync } from 'node:fs';

const [path, maxArg] = process.argv.slice(2);
const max = Number(maxArg ?? 16);
const g = onnx.ModelProto.decode(readFileSync(path)).graph;
for (const t of g.initializer) {
  const n = t.dims.reduce((a, d) => a * Number(d), 1);
  if (n > max) continue;
  const raw = t.rawData?.length ?? 0;
  console.log(`${t.name.padEnd(36)} dtype ${t.dataType} dims [${t.dims.join(',')}] n=${n}  raw ${raw}B float ${t.floatData?.length ?? 0} int64 ${t.int64Data?.length ?? 0} int32 ${t.int32Data?.length ?? 0} double ${t.doubleData?.length ?? 0}`);
}
