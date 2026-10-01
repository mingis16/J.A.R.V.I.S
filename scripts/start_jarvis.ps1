<#
Starts Alex's background services - trading bot, daemon, Telegram bot,
dashboard, voice - as hidden background processes, skipping any already running.
Output appends to logs\<name>_stdout.log. Safe to run repeatedly, e.g.
after a reboot.

    powershell -ExecutionPolicy Bypass -File scripts\start_jarvis.ps1          # start
    powershell -ExecutionPolicy Bypass -File scripts\start_jarvis.ps1 -Status  # show what's running
    powershell -ExecutionPolicy Bypass -File scripts\start_jarvis.ps1 -Stop    # stop everything
#>
param([switch]$Stop, [switch]$Status)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Logs = Join-Path $Root 'logs'
$PidFile = Join-Path $Root 'state\trading_bot.pid'

# Trading bot first: it attaches to (or launches) the MT5 terminal, and the
# daemon's first signal check connects to MT5 the moment it starts.
$Services = @(
    @{ Name = 'trading_bot'; Script = 'scripts\run_trading_bot.py';  Match = 'run_trading_bot\.py|trading_bot\.main' },
    @{ Name = 'daemon';      Script = 'scripts\run_daemon.py';       Match = 'run_daemon\.py' },
    @{ Name = 'telegram';    Script = 'scripts\run_telegram_bot.py'; Match = 'run_telegram_bot\.py' },
    @{ Name = 'dashboard';   Script = 'scripts\run_dashboard.py';    Match = 'run_dashboard\.py' },
    # Say "hey alex" to talk to it; replies are spoken aloud and transcripts land in logs\voice_stdout.log.
    @{ Name = 'voice';       Script = 'scripts\run_voice_assistant.py'; Match = 'run_voice_assistant\.py' }
)

function Get-ServiceProcesses($svc) {
    @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match $svc.Match })
}

function Show-Status {
    foreach ($svc in $Services) {
        $procs = Get-ServiceProcesses $svc
        if ($procs.Count -gt 0) {
            Write-Host ("{0,-12} running  (PID {1})" -f $svc.Name, (($procs | ForEach-Object ProcessId) -join ', '))
        } else {
            Write-Host ("{0,-12} STOPPED" -f $svc.Name)
        }
    }
}

if ($Status) { Show-Status; exit 0 }

if ($Stop) {
    foreach ($svc in $Services) {
        foreach ($p in Get-ServiceProcesses $svc) {
            & taskkill /PID $p.ProcessId /T /F | Out-Null
        }
    }
    # A deliberate stop, so don't leave a PID file for the daemon's health
    # check to report as "stopped unexpectedly".
    Remove-Item $PidFile -ErrorAction SilentlyContinue
    Show-Status
    exit 0
}

if (-not (Test-Path $Python)) { throw "Virtualenv python not found at $Python - run: python -m venv .venv; pip install -r requirements.txt" }
New-Item -ItemType Directory -Force $Logs | Out-Null

foreach ($svc in $Services) {
    $running = Get-ServiceProcesses $svc
    if ($running.Count -gt 0) {
        Write-Host "$($svc.Name): already running"
        continue
    }
    if ($svc.Name -eq 'trading_bot') {
        # Not running, so any PID file is stale (crash/reboot) and would make
        # the connect-wait below pass immediately.
        Remove-Item $PidFile -ErrorAction SilentlyContinue
    }
    $log = Join-Path $Logs "$($svc.Name)_stdout.log"
    $script = Join-Path $Root $svc.Script
    # cmd /c so stdout+stderr both append to the existing log (Start-Process's
    # own redirection truncates the file and can't merge the two streams).
    # -u: unbuffered output; -X utf8: emoji in messages can't hit cp1252 errors.
    $cmd = "`"`"$Python`" -u -X utf8 `"$script`" >> `"$log`" 2>&1`""
    Start-Process -FilePath 'cmd.exe' -ArgumentList "/c $cmd" -WorkingDirectory $Root -WindowStyle Hidden
    Write-Host "$($svc.Name): started (log: logs\$($svc.Name)_stdout.log)"

    if ($svc.Name -eq 'trading_bot') {
        # The bot writes its PID file only after it has connected to MT5, so
        # wait for that before starting other services that also connect.
        $deadline = (Get-Date).AddSeconds(90)
        while (-not (Test-Path $PidFile) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 2 }
        if (-not (Test-Path $PidFile)) {
            Write-Warning "Trading bot hasn't connected to MT5 after 90s - check logs\trading_bot_stdout.log. Starting the rest anyway."
        }
    }
}

# cmd.exe needs a moment to spawn python before the status check can see it.
Start-Sleep -Seconds 3
Write-Host ''
Show-Status
