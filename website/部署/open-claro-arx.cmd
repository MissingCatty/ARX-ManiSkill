@echo off
setlocal EnableExtensions DisableDelayedExpansion
title CLARO-ARX - SSH Access
set "CLARO_LAUNCHER_FILE=%~f0"
powershell.exe -NoLogo -NoProfile -Command "try { $text = [IO.File]::ReadAllText($env:CLARO_LAUNCHER_FILE); $parts = [regex]::Split($text, '(?m)^:: BEGIN_POWERSHELL\r?\n', 2); & ([scriptblock]::Create($parts[1])) } catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }"
set "CLARO_EXIT_CODE=%ERRORLEVEL%"
if not "%CLARO_EXIT_CODE%"=="0" (
    echo.
    echo Access could not be established or the SSH session ended with an error.
    pause
)
exit /b %CLARO_EXIT_CODE%
:: BEGIN_POWERSHELL
$ErrorActionPreference = 'Stop'
$remote = 'zhaoyechong@10.201.35.208'
$ssh = Get-Command ssh.exe -CommandType Application -ErrorAction SilentlyContinue
if (-not $ssh) {
    throw 'Windows OpenSSH (ssh.exe) is required.'
}

function Get-LocalEndpoint {
    foreach ($address in @([Net.IPAddress]::Loopback, [Net.IPAddress]::IPv6Loopback)) {
        $listener = [Net.Sockets.TcpListener]::new($address, 0)
        try {
            $listener.Start()
            $port = $listener.LocalEndpoint.Port
            $bind = if ($address.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetworkV6) {
                '[::1]'
            } else {
                '127.0.0.1'
            }
            return @{ Bind = $bind; Port = $port }
        } catch [Net.Sockets.SocketException] {
            continue
        } finally {
            $listener.Stop()
        }
    }
    throw 'Windows would not open an available local loopback port.'
}

function Test-Website($address) {
    try {
        $request = [Net.WebRequest]::Create($address + 'healthz')
        $request.Proxy = $null
        $request.Timeout = 1500
        $request.ReadWriteTimeout = 1500
        $request.AllowAutoRedirect = $false
        $response = $request.GetResponse()
        try {
            $reader = [IO.StreamReader]::new($response.GetResponseStream())
            try {
                return ([int]$response.StatusCode -eq 200 -and $reader.ReadToEnd().Trim() -eq 'ok')
            } finally {
                $reader.Dispose()
            }
        } finally {
            $response.Dispose()
        }
    } catch [Net.WebException] {
        return $false
    }
}

$local = Get-LocalEndpoint
$forward = '{0}:{1}:127.0.0.1:8786' -f $local.Bind, $local.Port
$url = 'http://{0}:{1}/' -f $local.Bind, $local.Port

Write-Host "Connecting to $remote"
Write-Host 'Complete any SSH login prompt in this window.'
Write-Host "Browser address: $url"
$sshArgs = @(
    '-N', '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
    '-L', $forward, $remote
)
$process = Start-Process -FilePath $ssh.Source -ArgumentList $sshArgs -NoNewWindow -PassThru
try {
    while (-not $process.HasExited) {
        if (Test-Website $url) {
            Write-Host 'Website ready. Keep this window open while using the site.'
            try {
                Start-Process $url
            } catch {
                Write-Host "Open $url in your browser."
            }
            break
        }
        Start-Sleep -Milliseconds 800
    }
    $process.WaitForExit()
    exit $process.ExitCode
} finally {
    if (-not $process.HasExited) {
        $process.Kill()
    }
    $process.Dispose()
}
