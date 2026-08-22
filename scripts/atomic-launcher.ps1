<#
.SYNOPSIS
    Canonical launcher for the Atomic llama.cpp fork.

.DESCRIPTION
    Resolves stack presets, validates fork-specific invariants, probes the selected
    llama-server, and emits or executes one exact argument vector. The JSON actions
    are the stable integration boundary used by llama-config-ui.
#>
[CmdletBinding()]
param(
    [ValidateSet('Describe', 'Resolve', 'Validate', 'Preview', 'Launch')]
    [string] $Action = 'Launch',
    [ValidateSet('gemma26', 'gemma12', 'qwen', 'ternary')]
    [string] $Stack = 'gemma26',
    [string] $Preset = 'standard',
    [string] $ConfigPath,
    [string] $ServerPath,
    [string] $ModelRoot,
    [switch] $SkipFileValidation,
    [switch] $NoCapabilityProbe,
    [switch] $Json,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArguments
)

$ErrorActionPreference = 'Stop'
$modulePath = Join-Path $PSScriptRoot 'atomic-launcher\AtomicLauncher.psm1'
$profilesPath = Join-Path $PSScriptRoot 'atomic-launcher\profiles.json'
Import-Module $modulePath -Force

function ConvertTo-LocalHashtable {
    param([object] $Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [pscustomobject]) {
        $result = @{}
        foreach ($property in $Value.PSObject.Properties) {
            $result[$property.Name] = ConvertTo-LocalHashtable $property.Value
        }
        return $result
    }
    if ($Value -is [System.Collections.IEnumerable] -and $Value -isnot [string]) {
        return @($Value | ForEach-Object { ConvertTo-LocalHashtable $_ })
    }
    return $Value
}

function Write-AtomicResult {
    param([object] $Value)
    if ($Json -or $Action -ne 'Launch') {
        $Value | ConvertTo-Json -Depth 20
    } else {
        $Value
    }
}

$overrides = @{}
if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) { throw "Configuration file not found: $ConfigPath" }
    $overrides = ConvertTo-LocalHashtable (Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json)
}
if (-not [string]::IsNullOrWhiteSpace($ServerPath)) { $overrides.server = $ServerPath }
if (-not [string]::IsNullOrWhiteSpace($ModelRoot)) { $overrides.model_root = $ModelRoot }
if ($ExtraArguments) { $overrides.extra_arguments = @($ExtraArguments) }

$configuration = Resolve-AtomicConfiguration -ProfilesPath $profilesPath -Stack $Stack -Preset $Preset -Overrides $overrides
$capabilities = $null
if (-not $NoCapabilityProbe -and (Test-Path -LiteralPath $configuration.server -PathType Leaf)) {
    $capabilities = Get-AtomicServerCapabilities -ServerPath $configuration.server
}

switch ($Action) {
    'Describe' {
        Write-AtomicResult ([pscustomobject] @{
            catalog = Get-AtomicProfileCatalog -ProfilesPath $profilesPath
            capabilities = $capabilities
            environment_features = @(
                [pscustomobject] @{ key = 'GGML_SCHED_PREFETCH_EXPERTS'; config = 'prefetch_experts'; supported = $true },
                [pscustomobject] @{ key = 'GGML_CUDA_REGISTER_HOST'; config = 'pin_host'; supported = $true },
                [pscustomobject] @{ key = 'TURBO_LAYER_ADAPTIVE'; config = 'turbo_layer_adaptive'; supported = $true }
            )
        })
        break
    }
    'Resolve' {
        Write-AtomicResult $configuration
        break
    }
    'Validate' {
        Write-AtomicResult (Test-AtomicConfiguration -Configuration $configuration -SkipFileValidation:$SkipFileValidation -Capabilities $capabilities)
        break
    }
    'Preview' {
        Write-AtomicResult (New-AtomicLaunchPlan -Configuration $configuration -SkipFileValidation:$SkipFileValidation -Capabilities $capabilities)
        break
    }
    'Launch' {
        $plan = New-AtomicLaunchPlan -Configuration $configuration -SkipFileValidation:$SkipFileValidation -Capabilities $capabilities
        if (-not $plan.validation.valid) {
            throw "Atomic launch validation failed:`n - $($plan.validation.errors -join "`n - ")"
        }
        foreach ($warning in $plan.validation.warnings) { Write-Warning $warning }
        # A long-lived terminal may retain settings from a previous launch. Clear
        # every environment feature owned by this launcher before applying the
        # selected plan so disabling a feature is deterministic.
        foreach ($name in @('GGML_SCHED_PREFETCH_EXPERTS', 'GGML_CUDA_REGISTER_HOST', 'TURBO_LAYER_ADAPTIVE')) {
            [Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
        foreach ($entry in $plan.environment.PSObject.Properties) {
            [Environment]::SetEnvironmentVariable($entry.Name, [string] $entry.Value, 'Process')
        }
        Write-Host ("Launching {0} ({1}) on {2}:{3}; spec={4}; vision={5}" -f `
            $configuration.stack_label, $configuration.preset, $configuration.host, $configuration.port,
            $(if ($configuration.spec_enabled) { $configuration.spec_type } else { 'none' }),
            [bool] $configuration.vision_enabled) -ForegroundColor Cyan
        & $plan.executable @($plan.arguments)
        exit $LASTEXITCODE
    }
}
