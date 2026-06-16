$listener = Get-NetTCPConnection -LocalPort 6379 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    Write-Output "Redis already listening on 6379"
    exit 0
}

$redisServer = Get-Command redis-server -ErrorAction SilentlyContinue
if (-not $redisServer) {
    Write-Error "redis-server was not found on PATH. Install Redis first."
    exit 1
}

Start-Process `
    -FilePath $redisServer.Source `
    -ArgumentList '--port 6379 --save "" --appendonly no' `
    -WorkingDirectory (Resolve-Path ".").Path `
    -WindowStyle Hidden

Start-Sleep -Seconds 2
$listener = Get-NetTCPConnection -LocalPort 6379 -State Listen -ErrorAction SilentlyContinue
if (-not $listener) {
    Write-Error "Redis did not start on port 6379"
    exit 1
}

Write-Output "Redis started on 6379"
