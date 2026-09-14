// Per-frequency-bin comparison of two [1,BINS,T] float32 dumps.
//
// 🔴 WHY. A whole-tensor relRMS is dominated by the loudest bins, so an operator that is WRONG in the
// quiet bins reads as 1e-6 and looks perfect - while every quiet bin is exactly where a phase (atan2) or a
// log-magnitude is decided. Splitting the statistic per bin is what turned "our STFT matches to 1.4e-6"
// into "bins 4 and 6 have the wrong sign".
import { readFileSync } from 'node:fs';
const f = p => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
const [ap, bp, binsStr] = process.argv.slice(2);
const a = f(ap), b = f(bp);
const BINS = parseInt(binsStr ?? '11', 10);
const T = a.length / BINS;
console.log('bin |    corr | relRMS  | rms(ours)  rms(ref)  | signDiff | first3 ours -> ref');
for (let k = 0; k < BINS; k++) {
  let aa = 0, bb = 0, ab = 0, dd = 0, dis = 0;
  for (let t = 0; t < T; t++) {
    const x = a[k * T + t], y = b[k * T + t];
    aa += x * x; bb += y * y; ab += x * y; dd += (x - y) ** 2;
    if (Math.sign(x) !== Math.sign(y)) dis++;
  }
  const s = [];
  for (let t = 0; t < 3; t++) s.push(`${a[k * T + t].toExponential(3)}->${b[k * T + t].toExponential(3)}`);
  console.log(`${String(k).padStart(3)} | ${(ab / Math.sqrt(aa * bb)).toFixed(5).padStart(8)} | `
            + `${Math.sqrt(dd / bb).toExponential(1)} | ${Math.sqrt(aa / T).toExponential(2)} ${Math.sqrt(bb / T).toExponential(2)} | `
            + `${String(dis).padStart(8)} | ${s.join('  ')}`);
}
