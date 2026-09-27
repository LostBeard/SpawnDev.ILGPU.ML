// Minimal static file server for the published Blazor Demo.
// COOP/COEP credentialless so SharedArrayBuffer / Wasm threads work (matches PMT).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../publish-demo-audit/wwwroot");
const PORT = Number(process.argv[3] || 5000);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".map": "application/json",
  ".txt": "text/plain",
  ".bin": "application/octet-stream",
  ".onnx": "application/octet-stream",
  ".tflite": "application/octet-stream",
  ".gguf": "application/octet-stream",
  ".wav": "audio/wav",
  ".mp3": "audio/mpeg",
  ".br": "application/octet-stream",
  ".gz": "application/octet-stream",
};

function send(res, status, body, headers = {}) {
  res.writeHead(status, {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "credentialless",
    ...headers,
  });
  res.end(body);
}

const server = http.createServer((req, res) => {
  try {
    let urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
    if (urlPath.endsWith("/")) urlPath += "index.html";
    let filePath = path.join(ROOT, urlPath);
    if (!filePath.startsWith(ROOT)) return send(res, 403, "forbidden");

    if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
      // SPA fallback for Blazor routes
      filePath = path.join(ROOT, "index.html");
    }
    if (!fs.existsSync(filePath)) return send(res, 404, "not found");

    const ext = path.extname(filePath).toLowerCase();
    const data = fs.readFileSync(filePath);
    send(res, 200, data, {
      "Content-Type": MIME[ext] || "application/octet-stream",
      "Cache-Control": "no-cache",
    });
  } catch (e) {
    send(res, 500, String(e));
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Serving ${ROOT}`);
  console.log(`http://127.0.0.1:${PORT}/  (COOP/COEP credentialless)`);
});
