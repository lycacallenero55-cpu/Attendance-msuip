<#
.SYNOPSIS
    Cleans up Python backend files that are no longer needed after migrating to Supabase.
.DESCRIPTION
    This script helps clean up Python backend files that are no longer needed
    after migrating to a serverless architecture with Supabase. It creates a backup
    of the backend directory before removing it.
.NOTES
    File Name      : cleanup_python_backend.ps1
    Prerequisite   : PowerShell 5.1 or later
#>

# Stop script execution when a non-terminating error occurs
$ErrorActionPreference = "Stop"

# Define paths
$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $projectRoot "attendance-backend"
$backupDir = Join-Path $projectRoot "attendance-backend-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

# Check if backend directory exists
if (-not (Test-Path $backendDir)) {
    Write-Host "Backend directory not found at: $backendDir" -ForegroundColor Yellow
    exit 0
}

# Create backup
Write-Host "Creating backup of backend directory at: $backupDir" -ForegroundColor Cyan
try {
    Copy-Item -Path $backendDir -Destination $backupDir -Recurse -Force
    Write-Host "Backup created successfully" -ForegroundColor Green
} catch {
    Write-Host "Failed to create backup: $_" -ForegroundColor Red
    exit 1
}

# List of files and directories to remove (relative to project root)
$itemsToRemove = @(
    "attendance-backend",
    ".github/workflows/backend-tests.yml",
    "Dockerfile.backend",
    "docker-compose.backend.yml",
    "requirements.txt",
    "alembic.ini"
)

# Remove files and directories
foreach ($item in $itemsToRemove) {
    $fullPath = Join-Path $projectRoot $item
    if (Test-Path $fullPath) {
        try {
            if ((Get-Item $fullPath) -is [System.IO.DirectoryInfo]) {
                Remove-Item -Path $fullPath -Recurse -Force
            } else {
                Remove-Item -Path $fullPath -Force
            }
            Write-Host "Removed: $item" -ForegroundColor Green
        } catch {
            Write-Host "Failed to remove $item : $_" -ForegroundColor Red
        }
    } else {
        Write-Host "Not found (skipping): $item" -ForegroundColor Yellow
    }
}

# Update package.json to remove backend-related scripts
$packageJsonPath = Join-Path $projectRoot "attendance-frontend\package.json"
if (Test-Path $packageJsonPath) {
    try {
        $packageJson = Get-Content $packageJsonPath -Raw | ConvertFrom-Json
        
        # Remove backend-related scripts
        $scriptsToRemove = @("start:backend", "dev:backend", "test:backend")
        foreach ($script in $scriptsToRemove) {
            if ($packageJson.scripts.PSObject.Properties.Name -contains $script) {
                $packageJson.scripts.PSObject.Properties.Remove($script)
                Write-Host "Removed script: $script" -ForegroundColor Green
            }
        }
        
        # Save the updated package.json
        $packageJson | ConvertTo-Json -Depth 10 | Set-Content -Path $packageJsonPath -Encoding UTF8
        Write-Host "Updated package.json" -ForegroundColor Green
    } catch {
        Write-Host "Failed to update package.json: $_" -ForegroundColor Red
    }
}

# Update README.md to remove backend-specific instructions
$readmePath = Join-Path $projectRoot "README.md"
if (Test-Path $readmePath) {
    try {
        $readmeContent = Get-Content $readmePath -Raw
        # Remove backend-specific sections if they exist
        $readmeContent = $readmeContent -replace '(?s)## Backend Development.*?(?=##|$)', ''
        $readmeContent = $readmeContent -replace '(?s)### Database Migrations.*?(?=###|$)', ''
        
        # Save the updated README
        $readmeContent | Set-Content -Path $readmePath -Encoding UTF8
        Write-Host "Updated README.md" -ForegroundColor Green
    } catch {
        Write-Host "Failed to update README.md: $_" -ForegroundColor Red
    }
}

Write-Host "\nCleanup complete!" -ForegroundColor Cyan
Write-Host "A backup of the backend has been saved to: $backupDir" -ForegroundColor Cyan
Write-Host "You can delete this backup once you've verified everything is working correctly." -ForegroundColor Cyan
