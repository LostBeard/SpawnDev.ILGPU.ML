// Static file server with CORS and byte ranges, for running the browser-lane SkinTokens test against local weights / goldens.
//   node serve-files.mjs <dir> [port=8765]
// Serves <dir> read-only on 127.0.0.1 only. Kill it by the PID that started it when the run is over.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(process.argv[2] ?? '.');
const port = Number(process.argv[3] ?? 8765);
const cors = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*', 'Access-Control-Expose-Headers': 'Content-Length, Content-Range, Accept-Ranges', 'Accept-Ranges': 'bytes' };

http.createServer((req, res) => {
    if (req.method === 'OPTIONS') { res.writeHead(204, cors); return res.end(); }
    const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    if (!file.startsWith(root) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { res.writeHead(404, cors); return res.end('not found'); }
    const size = fs.statSync(file).size;
    const m = /bytes=(\d*)-(\d*)/.exec(req.headers.range ?? '');
    let start = 0, end = size - 1, status = 200;
    if (m) { start = m[1] === '' ? size - Number(m[2]) : Number(m[1]); end = m[2] === '' || m[1] === '' ? size - 1 : Math.min(Number(m[2]), size - 1); status = 206; }
    const h = { ...cors, 'Content-Type': 'application/octet-stream', 'Content-Length': end - start + 1 };
    if (status === 206) h['Content-Range'] = `bytes ${start}-${end}/${size}`;
    res.writeHead(status, h);
    if (req.method === 'HEAD') return res.end();
    fs.createReadStream(file, { start, end }).pipe(res);
}).listen(port, '127.0.0.1', () => console.log(`serving ${root} on http://127.0.0.1:${port}/`));
