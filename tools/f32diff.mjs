// Compare two raw float32 files: correlation, relative RMS, best-fit scale.
//
// 🔴 WHY. Before calling OUR output wrong, the ORACLE has to be shown to be stable. Adding graph outputs
// changes which fusions onnxruntime applies, so two onnxruntime runs of the SAME model are not bit-equal.
// If two reference runs already disagree by X, then X is the floor and chasing below it is chasing noise.
import { readFileSync } from 'node:fs';
const f = p => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
const [ap, bp] = process.argv.slice(2);
const a = f(ap), b = f(bp);
const n = Math.min(a.length, b.length);
let aa = 0, bb = 0, ab = 0, dd = 0, mx = 0;
for (let i = 0; i < n; i++) {
  const d = a[i] - b[i];
  aa += a[i] * a[i]; bb += b[i] * b[i]; ab += a[i] * b[i]; dd += d * d;
  if (Math.abs(d) > mx) mx = Math.abs(d);
}
console.log(`n=${n} (${a.length} vs ${b.length}) corr=${(ab / Math.sqrt(aa * bb)).toFixed(6)} `
          + `relRMS=${Math.sqrt(dd / bb).toExponential(2)} max|diff|=${mx.toExponential(2)} scale=${(ab / bb).toFixed(4)}`);
