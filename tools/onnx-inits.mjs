import onnxProto from 'onnx-proto';
const { onnx } = onnxProto;
import { readFileSync } from 'node:fs';
const m = onnx.ModelProto.decode(readFileSync(process.argv[2]));
const counts = new Map();
for (const t of m.graph.initializer) {
  const k = Object.entries(onnx.TensorProto.DataType).find(([,v]) => v === t.dataType)?.[0] ?? t.dataType;
  counts.set(k, (counts.get(k) ?? 0) + 1);
}
console.log('initializer dtypes:', [...counts].map(([k,v]) => `${k}=${v}`).join(' '));
for (const n of ['/encoder/F0.2/Mul_1_output_0','/encoder/F0.2/Mul_output_0','encoder.predictor.F0.2.conv1.bias']) {
  const t = m.graph.initializer.find(t => t.name === n);
  console.log(n, t ? `dtype=${t.dataType} dims=${t.dims.join('x')} raw=${(t.rawData?.length)??0}` : 'MISSING');
}
console.log('opset', m.opsetImport.map(o => `${o.domain||'ai.onnx'}:${o.version}`).join(' '));
