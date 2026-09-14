// Zero specific bins of a [1,BINS,T] dump. For the STFT of a REAL signal the DC and Nyquist bins have an
// exactly zero imaginary part; any residue there is the transform's own rounding, and it decides a phase.
import { readFileSync, writeFileSync } from 'node:fs';
const [src, dst, binsCsv, binsStr] = process.argv.slice(2);
const b = readFileSync(src);
const a = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
const BINS = parseInt(binsStr ?? '11', 10), T = a.length / BINS;
for (const k of binsCsv.split(',').map(Number)) for (let t = 0; t < T; t++) a[k * T + t] = 0;
writeFileSync(dst, Buffer.from(a.buffer));
console.log(`zeroed bins [${binsCsv}] of ${BINS}, T=${T}`);
