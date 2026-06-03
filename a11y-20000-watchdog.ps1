$ErrorActionPreference = "Stop"

$IdentityFile = Join-Path $env:USERPROFILE ".ssh\autodl_vdroid"

$SshArgs = @(
  "-F", "NUL",
  "-p", "18563",
  "-N",
  "-v",
  "-o", "ExitOnForwardFailure=yes",
  "-o", "ServerAliveInterval=30",
  "-o", "ServerAliveCountMax=3",
  "-L", "20000:127.0.0.1:20000",
  "root@connect.westc.seetacloud.com"
)

if (Test-Path $IdentityFile) {
  $SshArgs = @(
    "-F", "NUL",
    "-p", "18563",
    "-N",
    "-v",
    "-i", $IdentityFile,
    "-o", "BatchMode=yes",
    "-o", "PreferredAuthentications=publickey",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-L", "20000:127.0.0.1:20000",
    "root@connect.westc.seetacloud.com"
  )
  Write-Host "Using SSH key: $IdentityFile"
} else {
  Write-Warning "SSH key not found: $IdentityFile. Password prompts may appear after restart."
}

while ($true) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $stdout = Join-Path $env:TEMP "a11y-20000-$stamp.out.log"
  $stderr = Join-Path $env:TEMP "a11y-20000-$stamp.err.log"

  Write-Host "Starting a11y 20000 forwarder..."
  $process = Start-Process -FilePath "ssh.exe" `
    -ArgumentList $SshArgs `
    -NoNewWindow `
    -PassThru `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr

  while (-not $process.HasExited) {
    Start-Sleep -Seconds 2

    if ((Test-Path $stderr) -and (Select-String -Path $stderr -Pattern "accept: Too many open files" -Quiet)) {
      Write-Warning "Detected 'accept: Too many open files'. Restarting 20000 forwarder..."
      Stop-Process -Id $process.Id -Force
      break
    }
  }

  Write-Host "a11y 20000 forwarder stopped. Restarting in 3 seconds..."
  Start-Sleep -Seconds 3
}
