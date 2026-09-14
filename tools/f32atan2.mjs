import { readFileSync, writeFileSync } from 'node:fs';
const f = p => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
const [imP, reP, outP] = process.argv.slice(2);
const im = f(imP), re = f(reP);
const out = new Float32Array(im.length);
for (let i = 0; i < im.length; i++) out[i] = Math.fround(Math.atan2(im[i], re[i]));
writeFileSync(outP, Buffer.from(out.buffer));
console.log(`atan2 of ${im.length} values`);
