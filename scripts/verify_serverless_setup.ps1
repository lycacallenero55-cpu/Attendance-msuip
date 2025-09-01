<#
.SYNOPSIS
    Verifies the serverless setup of the Attendance Monitoring System.
.DESCRIPTION
    This script verifies that all components of the serverless architecture
    are properly configured and working with Supabase.
.NOTES
    File Name      : verify_serverless_setup.ps1
    Prerequisite   : PowerShell 5.1 or later, Node.js, npm
#>

# Stop script execution when a non-terminating error occurs
$ErrorActionPreference = "Stop"

# Define paths
$projectRoot = Split-Path -Parent $PSScriptRoot
$frontendDir = Join-Path $projectRoot "attendance-frontend"
$envFile = Join-Path $frontendDir ".env"

# Check if Node.js and npm are installed
try {
    $nodeVersion = node --version
    $npmVersion = npm --version
    Write-Host "Node.js version: $nodeVersion" -ForegroundColor Green
    Write-Host "npm version: $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "Node.js and npm are required to run this script" -ForegroundColor Red
    exit 1
}

# Check if .env file exists
if (-not (Test-Path $envFile)) {
    Write-Host "Error: .env file not found in $frontendDir" -ForegroundColor Red
    Write-Host "Please create a .env file with your Supabase credentials" -ForegroundColor Yellow
    Write-Host "Example:"
    Write-Host "VITE_SUPABASE_URL=your-project-url"
    Write-Host "VITE_SUPABASE_ANON_KEY=your-anon-key"
    exit 1
}

# Load environment variables
$envContent = Get-Content $envFile -Raw
$envVars = @{}

# Parse .env file
$envContent -split "`n" | ForEach-Object {
    if ($_ -match '^([^#][^=]+)=(.*)') {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim('"', '\', ' ')
        $envVars[$key] = $value
    }
}

# Check required environment variables
$requiredVars = @("VITE_SUPABASE_URL", "VITE_SUPABASE_ANON_KEY")
$missingVars = $requiredVars | Where-Object { -not $envVars.ContainsKey($_) }

if ($missingVars) {
    Write-Host "Error: Missing required environment variables:" -ForegroundColor Red
    $missingVars | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

# Install dependencies if node_modules doesn't exist
$nodeModulesDir = Join-Path $frontendDir "node_modules"
if (-not (Test-Path $nodeModulesDir)) {
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    Set-Location $frontendDir
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

# Run TypeScript type checking
Write-Host "`nRunning TypeScript type checking..." -ForegroundColor Cyan
Set-Location $frontendDir
npm run typecheck
if ($LASTEXITCODE -ne 0) {
    Write-Host "TypeScript type checking failed" -ForegroundColor Red
    exit 1
}
Write-Host "TypeScript type checking passed" -ForegroundColor Green

# Run tests
Write-Host "`nRunning tests..." -ForegroundColor Cyan
npm test -- --watchAll=false
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed" -ForegroundColor Red
    exit 1
}
Write-Host "All tests passed" -ForegroundColor Green

# Verify Supabase connection
Write-Host "`nVerifying Supabase connection..." -ForegroundColor Cyan
$supabaseTestScript = @"
const { createClient } = require('@supabase/supabase-js');

async function testSupabaseConnection() {
  const supabaseUrl = process.env.VITE_SUPABASE_URL;
  const supabaseKey = process.env.VITE_SUPABASE_ANON_KEY;
  
  if (!supabaseUrl || !supabaseKey) {
    console.error('Missing Supabase credentials');
    process.exit(1);
  }

  const supabase = createClient(supabaseUrl, supabaseKey);
  
  try {
    // Test auth
    console.log('Testing authentication...');
    const { data: authData, error: authError } = await supabase.auth.signInAnonymously();
    
    if (authError) {
      if (authError.message.includes('Signups not allowed for anonymous')) {
        console.log('  ✓ Anonymous auth is properly restricted (expected)');
      } else {
        throw authError;
      }
    } else {
      console.log('  ✓ Anonymous auth successful');
    }
    
    // Test database access
    console.log('Testing database access...');
    const { data: sessions, error: dbError } = await supabase
      .from('sessions')
      .select('*')
      .limit(1);
      
    if (dbError) {
      if (dbError.message.includes('permission denied')) {
        console.log('  ✓ Database access is properly restricted (expected)');
      } else {
        throw dbError;
      }
    } else {
      console.log('  ✓ Database access successful');
    }
    
    console.log('✅ Supabase connection test passed');
    process.exit(0);
  } catch (error) {
    console.error('❌ Supabase test failed:', error);
    process.exit(1);
  }
}

testSupabaseConnection();
"@

$tempScriptPath = [System.IO.Path]::GetTempFileName() + ".js"
$supabaseTestScript | Out-File -FilePath $tempScriptPath -Encoding utf8

# Set environment variables for the test script
$env:VITE_SUPABASE_URL = $envVars["VITE_SUPABASE_URL"]
$env:VITE_SUPABASE_ANON_KEY = $envVars["VITE_SUPABASE_ANON_KEY"]

try {
    $supabaseTestOutput = node $tempScriptPath 2>&1 | Out-String
    Write-Host $supabaseTestOutput -ForegroundColor Green
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Supabase connection test failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Supabase connection test passed" -ForegroundColor Green
} finally {
    # Clean up
    if (Test-Path $tempScriptPath) {
        Remove-Item $tempScriptPath -Force
    }
}

# Verify frontend build
Write-Host "`nVerifying frontend build..." -ForegroundColor Cyan
Set-Location $frontendDir
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Frontend build failed" -ForegroundColor Red
    exit 1
}
Write-Host "Frontend build successful" -ForegroundColor Green

# Check for any remaining backend files
$backendFiles = @(
    "attendance-backend",
    "Dockerfile.backend",
    "docker-compose.backend.yml",
    "requirements.txt",
    "alembic.ini"
)

$foundBackendFiles = $backendFiles | Where-Object { Test-Path (Join-Path $projectRoot $_) }

if ($foundBackendFiles) {
    Write-Host "`n⚠️  Found backend files that can be removed:" -ForegroundColor Yellow
    $foundBackendFiles | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    Write-Host "Run .\scripts\cleanup_python_backend.ps1 to remove these files" -ForegroundColor Yellow
} else {
    Write-Host "`n✅ No backend files found" -ForegroundColor Green
}

Write-Host "`n🎉 Serverless setup verification complete!" -ForegroundColor Green
Write-Host "Your application is ready for deployment to a static hosting provider." -ForegroundColor Cyan
Write-Host "Recommended next steps:" -ForegroundColor Cyan
Write-Host "1. Commit your changes to version control" -ForegroundColor Cyan
Write-Host "2. Deploy to Vercel, Netlify, or your preferred static hosting" -ForegroundColor Cyan
Write-Host "3. Set up environment variables in your hosting provider" -ForegroundColor Cyan
