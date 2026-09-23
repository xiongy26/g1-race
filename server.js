// 极简静态文件服务器(node server.js [port])
const http = require('http');
const fs = require('fs');
const path = require('path');

const root = __dirname;
const port = parseInt(process.argv[2] || '8137', 10);
const mime = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.stl': 'application/octet-stream',
  '.xml': 'application/xml', '.json': 'application/json', '.css': 'text/css',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.map': 'application/json',
};

http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p.endsWith('/')) p += 'index.html';
  const file = path.normalize(path.join(root, p));
  if (!file.startsWith(root)) { res.writeHead(403); res.end(); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('404'); return; }
    res.writeHead(200, {
      'Content-Type': mime[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    res.end(data);
  });
}).listen(port, '127.0.0.1', () => console.log(`serving ${root} at http://127.0.0.1:${port}`));
