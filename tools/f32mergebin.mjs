// Copy selected bins from `donor` into `base` (both [1,BINS,T]) - "what would the output be if JUST these
// channels matched?", to size one channel's contribution without changing anything else.
import { readFileSync, writeFileSync } from 'node:fs';
const [basePath, donorPath, dst, binsCsv, binsStr] = process.argv.slice(2);
const bb = readFileSync(basePath);
const a = new Float32Array(bb.buffer.slice(bb.byteOffset, bb.byteOffset + bb.length));
const db = readFileSync(donorPath);
const d = new Float32Array(db.buffer, db.byteOffset, db.length / 4);
const BINS = parseInt(binsStr ?? '11', 10), T = a.length / BINS;
for (const k of binsCsv.split(',').map(Number)) for (let t = 0; t < T; t++) a[k * T + t] = d[k * T + t];
writeFileSync(dst, Buffer.from(a.buffer));
console.log(`merged bins [${binsCsv}] from donor`);
