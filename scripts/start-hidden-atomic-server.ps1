param(
    [ValidateSet('Ternary', 'Gemma', 'Qwen')]
    [string] $Stack = 'Ternary',
    [string] $Server = (Join-Path $PSScriptRoot '..\build-rollover\bin\Release\llama-server.exe'),
    [int] $ContextSize = 32768,
    [int] $Port = 8097,
    [string] $LogDirectory = (Join-Path $PSScriptRoot '..\build-rollover\validation'),
    [int] $ReadyTimeoutSeconds = 60,
    # Directory that holds the stack's GGUF files. The default keeps the
    # installed-binaries layout; pass the current model directory explicitly.
    [string] $ModelRoot = 'C:\Program Files\llamacpp'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ModelRoot -PathType Container)) {
    throw "Model root does not exist: $ModelRoot"
}
$modelRoot = $ModelRoot
$common = @(
    '--ctx-size', $ContextSize,
    '--n-gpu-layers', 999,
    '--keep', -1,
    '--parallel', 1,
    '--host', '127.0.0.1',
    '--port', $Port,
    '--load-mode', 'mlock',
    '--jinja',
    '--flash-attn', 'on',
    '--context-shift',
    '--metrics'
)

$environment = @{}
switch ($Stack) {
    'Ternary' {
        $model = Join-Path $modelRoot 'Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf'
        $draft = Join-Path $modelRoot 'Ternary-Bonsai-27B-dspark-Q4_1.gguf'
        $mmproj = Join-Path $modelRoot 'Ternary-Bonsai-27B-mmproj-Q8_0.gguf'
        $arguments = @(
            '--model', $model,
            '--cache-type-k', 'turbo4', '--cache-type-v', 'turbo3',
            '--mmproj', $mmproj,
            '--spec-type', 'draft-dspark', '--model-draft', $draft,
            '--spec-draft-ngl', 999,
            '--spec-draft-type-k', 'q8_0', '--spec-draft-type-v', 'q8_0',
            '--spec-draft-n-max', 4,
            '--reasoning', 'on'
        ) + $common
    }
    'Gemma' {
        $model = Join-Path $modelRoot 'gemma-4-26B-A4B-it-qat-q4_0-unquantized-heretic.i1-Q4_K_M.gguf'
        $draft = Join-Path $modelRoot 'mtp-gemma-4-26B-A4B-it.gguf'
        $mmproj = Join-Path $modelRoot 'mmproj-gemma-4-26B-A4B-it-qat-heretic-BF16.gguf'
        $arguments = @(
            '--model', $model,
            '--cache-type-k', 'turbo4', '--cache-type-v', 'turbo2',
            '--n-cpu-moe', 14,
            '--mmproj', $mmproj,
            '--spec-type', 'draft-mtp', '--model-draft', $draft,
            '--spec-draft-ngl', 999,
            '--spec-draft-type-k', 'turbo3', '--spec-draft-type-v', 'turbo3',
            '--spec-draft-n-max', 2,
            '--reasoning', 'off'
        ) + $common
        $environment.GGML_SCHED_PREFETCH_EXPERTS = '1'
    }
    'Qwen' {
        $model = Join-Path $modelRoot 'Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-I-Compact.gguf'
        $mmproj = Join-Path $modelRoot 'mmproj-Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-F16.gguf'
        $arguments = @(
            '--model', $model,
            '--cache-type-k', 'turbo4', '--cache-type-v', 'turbo3',
            '--n-cpu-moe', 28,
            '--mmproj', $mmproj,
            '--spec-type', 'draft-mtp',
            '--spec-draft-ngl', 999,
            '--spec-draft-type-k', 'q8_0', '--spec-draft-type-v', 'q8_0',
            '--spec-draft-n-max', 2,
            '--reasoning', 'off'
        ) + $common
        $environment.GGML_SCHED_PREFETCH_EXPERTS = '1'
    }
}

foreach ($path in @($Server, $model, $draft, $mmproj)) {
    if ($path -and -not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required $Stack validation artifact does not exist: $path"
    }
}

$null = New-Item -ItemType Directory -Force -Path $LogDirectory
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$prefix = $Stack.ToLowerInvariant()
$stdout = Join-Path $LogDirectory "$prefix-$stamp.stdout.log"
$stderr = Join-Path $LogDirectory "$prefix-$stamp.stderr.log"

# Start-Process joins ArgumentList values into a command line on Windows. Quote
# only values containing whitespace so paths remain single native argv entries.
$nativeArguments = foreach ($argument in $arguments) {
    $value = [string] $argument
    if ($value -match '\s') { '"' + $value.Replace('"', '\"') + '"' } else { $value }
}

# Keep exact-model validation detached from the controlling Codex process.
# Backend faults have previously terminated an interactive controller; explicit
# files preserve complete evidence and the caller always uses a finite deadline.
$process = Start-Process `
    -FilePath (Resolve-Path -LiteralPath $Server) `
    -ArgumentList $nativeArguments `
    -Environment $environment `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$ready = $false
$deadline = [DateTime]::UtcNow.AddSeconds($ReadyTimeoutSeconds)
while (-not $process.HasExited -and [DateTime]::UtcNow -lt $deadline) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        if ($health.status -eq 'ok') {
            $ready = $true
            break
        }
    } catch {
        # Loading is expected to refuse connections until initialization ends.
    }
    Start-Sleep -Milliseconds 500
}

if (-not $ready) {
    # A readiness timeout must not strand a VRAM-heavy server. Preserve the
    # redirected logs, then tear down only the child this invocation created.
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit()
    }
}

[pscustomobject]@{
    process_id = $process.Id
    ready = $ready
    exited = $process.HasExited
    exit_code = if ($process.HasExited) { $process.ExitCode } else { $null }
    stack = $Stack
    port = $Port
    context_size = $ContextSize
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json -Compress

if (-not $ready) {
    exit 1
}
