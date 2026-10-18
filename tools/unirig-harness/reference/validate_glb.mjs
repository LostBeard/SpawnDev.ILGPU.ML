import fs from 'node:fs';
import validator from 'gltf-validator';
const buf = fs.readFileSync(process.argv[2]);
const r = await validator.validateBytes(new Uint8Array(buf), { maxIssues: 50 });
const i = r.issues;
console.log(`validator ${r.validatorVersion}: errors ${i.numErrors}, warnings ${i.numWarnings}, infos ${i.numInfos}, hints ${i.numHints}`);
for (const m of i.messages.filter(m => m.severity <= 1).slice(0, 12)) console.log(`  [${['ERROR', 'WARNING'][m.severity]}] ${m.code} ${m.pointer} - ${m.message}`);
const info = r.info;
console.log(`  skins ${info.skinsCount ?? '?'}, joints max ${info.maxJoints ?? '?'}, animations ${info.animationCount}, drawCalls ${info.drawCallCount}, vertices ${info.totalVertexCount}, triangles ${info.totalTriangleCount}`);
