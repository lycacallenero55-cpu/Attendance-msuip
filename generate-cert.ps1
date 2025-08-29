# Create certs directory if it doesn't exist
$certsDir = "certs"
if (-not (Test-Path -Path $certsDir)) {
    New-Item -ItemType Directory -Path $certsDir | Out-Null
}

# Generate private key and self-signed certificate
openssl req -x509 \
    -newkey rsa:4096 \
    -keyout "$certsDir/localhost-key.pem" \
    -out "$certsDir/localhost.pem" \
    -sha256 \
    -days 3650 \
    -nodes \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

Write-Host "Certificate files have been generated in the 'certs' directory."
