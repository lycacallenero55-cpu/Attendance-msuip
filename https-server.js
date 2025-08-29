const https = require('https');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const selfsigned = require('selfsigned');

// Generate self-signed certificate
const attrs = [{ name: 'commonName', value: 'localhost' }];
const pems = selfsigned.generate(attrs, { days: 365 });

const options = {
  key: pems.private,
  cert: pems.cert
};

// Create HTTPS server
const server = https.createServer(options, (req, res) => {
  // Proxy the request to the Vite dev server
  const proxy = https.request({
    hostname: 'localhost',
    port: 8080,
    path: req.url,
    method: req.method,
    headers: req.headers,
    rejectUnauthorized: false // For self-signed certificate
  }, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  req.pipe(proxy, { end: true });
});

// Start the server
const PORT = 8443;
server.listen(PORT, '0.0.0.0', () => {
  console.log(`HTTPS server running at https://localhost:${PORT}`);
  console.log(`Access from other devices: https://${getLocalIpAddress()}:${PORT}`);
});

// Get local IP address
function getLocalIpAddress() {
  const { networkInterfaces } = require('os');
  const nets = networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      // Skip over non-IPv4 and internal (i.e. 127.0.0.1) addresses
      if (net.family === 'IPv4' && !net.internal) {
        return net.address;
      }
    }
  }
  return 'localhost';
}

// Start Vite dev server
console.log('Starting Vite dev server...');
const vite = exec('npm run dev', { stdio: 'inherit' });

// Handle process termination
process.on('SIGINT', () => {
  console.log('Shutting down servers...');
  server.close();
  vite.kill();
  process.exit();
});
