[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $ServerPath,

    [Parameter(Mandatory = $true)]
    [string] $ModelPath,

    [string] $DraftModelPath,

    [string] $MmprojPath,

    [string] $ImagePath,

    [switch] $TargetOnly,
    [switch] $TextOnly,
    [ValidateSet('draft-dspark', 'draft-dflash', 'draft-mtp', 'draft-simple', 'draft-eagle3',
        'ngram-simple', 'ngram-map-k', 'ngram-map-k4v', 'ngram-mod', 'ngram-cache')]
    [string] $SpecType = 'draft-dspark',
    [double] $SpecDraftPMin = -1.0,
    [int] $SpecDraftNMax = 4,
    [int] $DraftGpuLayers = 999,
    [int] $CpuMoe = 0,
    [string] $FollowUpPrompt = 'What command name is highlighted in the terminal? Answer with only the command.',
    [int] $FirstMaxTokens = 32,
    [int] $SecondMaxTokens = 16,
    [int] $Port = 18091,
    [int] $StartupTimeoutSeconds = 420,
    [int] $RequestTimeoutSeconds = 300
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$requiredPaths = @($ServerPath, $ModelPath)
if (-not $TextOnly) {
    if ([string]::IsNullOrWhiteSpace($MmprojPath) -or [string]::IsNullOrWhiteSpace($ImagePath)) {
        throw 'MmprojPath and ImagePath are required unless TextOnly is selected.'
    }
    $requiredPaths += @($MmprojPath, $ImagePath)
} elseif (-not [string]::IsNullOrWhiteSpace($MmprojPath)) {
    # Loading an mmproj during a text-only request proves that real media, not
    # the mere presence of the vision projector, controls speculative policy.
    $requiredPaths += $MmprojPath
}
if (-not $TargetOnly -and $SpecType -in @('draft-dspark', 'draft-dflash', 'draft-simple', 'draft-eagle3')) {
    if ([string]::IsNullOrWhiteSpace($DraftModelPath)) {
        throw 'DraftModelPath is required unless TargetOnly is selected.'
    }
    $requiredPaths += $DraftModelPath
}
foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file does not exist: $path"
    }
}

$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('atomic-media-spec-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
$stdoutPath = Join-Path $testRoot 'server.stdout.log'
$stderrPath = Join-Path $testRoot 'server.stderr.log'

function ConvertTo-NativeQuotedArgument {
    param([string] $Value)

    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    # Start-Process joins ArgumentList into one native command line. The model
    # paths used here do not contain literal quotes, but reject them explicitly
    # so a diagnostic helper never becomes a command-line injection surface.
    if ($Value.Contains('"')) {
        throw 'Native test arguments cannot contain a quote character.'
    }
    return '"' + $Value + '"'
}

$arguments = @(
    '--model', $ModelPath,
    '--ctx-size', '8192',
    '--n-gpu-layers', '999',
    '--keep', '-1',
    '--parallel', '1',
    '--host', '127.0.0.1',
    '--port', [string] $Port,
    '--cache-type-k', 'turbo4',
    '--cache-type-v', 'turbo3',
    '--load-mode', 'mlock',
    '--jinja',
    '--flash-attn', 'on',
    '--context-shift',
    '--reasoning', 'off',
    '--metrics'
)
if (-not [string]::IsNullOrWhiteSpace($MmprojPath)) {
    $arguments += @('--mmproj', $MmprojPath)
}
if ($CpuMoe -gt 0) {
    $arguments += @('--n-cpu-moe', [string] $CpuMoe)
}
if (-not $TargetOnly) {
    $arguments += @('--spec-type', $SpecType, '--spec-draft-n-max', [string] $SpecDraftNMax)
    if ($SpecDraftPMin -ge 0.0) {
        $arguments += @('--spec-draft-p-min', $SpecDraftPMin.ToString([Globalization.CultureInfo]::InvariantCulture))
    }
    if ($SpecType -in @('draft-dspark', 'draft-dflash', 'draft-mtp', 'draft-simple', 'draft-eagle3')) {
        $arguments += @('--spec-draft-type-k', 'q8_0', '--spec-draft-type-v', 'q8_0')
    }
    if (-not [string]::IsNullOrWhiteSpace($DraftModelPath)) {
        $arguments += @(
            '--model-draft', $DraftModelPath,
            '--spec-draft-ngl', [string] $DraftGpuLayers
        )
    }
}
$nativeArguments = (($arguments | ForEach-Object {
    ConvertTo-NativeQuotedArgument ([string] $_)
}) -join ' ')

$process = Start-Process `
    -FilePath (Resolve-Path -LiteralPath $ServerPath).Path `
    -ArgumentList $nativeArguments `
    -WorkingDirectory (Split-Path -Parent (Resolve-Path -LiteralPath $ServerPath).Path) `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

try {
    $baseUri = "http://127.0.0.1:$Port"
    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    $ready = $false

    while ([DateTime]::UtcNow -lt $deadline) {
        $process.Refresh()
        if ($process.HasExited) {
            throw "Server exited before readiness with code $($process.ExitCode). Logs: $testRoot"
        }

        try {
            $health = Invoke-RestMethod -Uri "$baseUri/health" -TimeoutSec 2
            if ($health.status -eq 'ok') {
                $ready = $true
                break
            }
        } catch {
            # Loading a multi-gigabyte target/draft pair is expected to outlive
            # several health-probe timeouts.
        }
        Start-Sleep -Seconds 1
    }

    if (-not $ready) {
        throw "Server did not become ready within $StartupTimeoutSeconds seconds. Logs: $testRoot"
    }

    if ($TextOnly) {
        $firstUserContent = 'Briefly explain why speculative decoding verifies draft tokens with the target model.'
    } else {
        $imageBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($ImagePath))
        $imageUrl = 'data:image/png;base64,' + $imageBase64
        $firstUserContent = @(
            @{ type = 'text'; text = 'Briefly describe what is visible in this screenshot.' },
            @{ type = 'image_url'; image_url = @{ url = $imageUrl } }
        )
    }

    $request1 = @{
        model = 'default'
        messages = @(@{ role = 'user'; content = $firstUserContent })
        temperature = 0.0
        top_k = 1
        seed = 4242
        max_tokens = $FirstMaxTokens
        stream = $false
        logprobs = $true
        top_logprobs = 1
    }
    $response1 = Invoke-RestMethod `
        -Method Post `
        -Uri "$baseUri/v1/chat/completions" `
        -ContentType 'application/json' `
        -Body ($request1 | ConvertTo-Json -Depth 12 -Compress) `
        -TimeoutSec $RequestTimeoutSeconds
    $answer1 = [string] $response1.choices[0].message.content

    # This deliberately resends the original image, matching normal stateless
    # OpenAI-compatible clients and reproducing the former permanent-disable bug.
    $request2 = @{
        model = 'default'
        messages = @(
            @{ role = 'user'; content = $firstUserContent },
            @{ role = 'assistant'; content = $answer1 },
            @{ role = 'user'; content = $FollowUpPrompt }
        )
        temperature = 0.0
        top_k = 1
        seed = 4242
        max_tokens = $SecondMaxTokens
        stream = $false
        logprobs = $true
        top_logprobs = 1
    }
    $response2 = Invoke-RestMethod `
        -Method Post `
        -Uri "$baseUri/v1/chat/completions" `
        -ContentType 'application/json' `
        -Body ($request2 | ConvertTo-Json -Depth 12 -Compress) `
        -TimeoutSec $RequestTimeoutSeconds
    $answer2 = [string] $response2.choices[0].message.content

    # Preserve the complete API evidence beside the server logs. Content alone
    # can hide token-boundary differences, while logprobs expose the exact
    # committed token sequence used by the deterministic parity oracle.
    $response1 | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $testRoot 'response-1.json')
    $response2 | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $testRoot 'response-2.json')

    [pscustomobject] @{
        TestRoot = $testRoot
        Mode = if ($TargetOnly) { 'target-only' } else { $SpecType }
        RequestKind = if ($TextOnly) { 'text-only' } else { 'media-history' }
        ProcessId = $process.Id
        FirstAnswer = $answer1
        SecondAnswer = $answer2
        FirstPromptTokens = $response1.usage.prompt_tokens
        SecondPromptTokens = $response2.usage.prompt_tokens
    } | ConvertTo-Json -Compress
} finally {
    $process.Refresh()
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit(10000) | Out-Null
    }
}
