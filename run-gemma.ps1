<#
    Owner launcher for the Atomic llama.cpp fork.

    This tracked compatibility wrapper translates the owner's established short
    commands into the canonical launcher. Installed model locations remain catalog
    data; the wrapper does not duplicate model-stack or validation logic.

    Normal configurations:
      .\run-gemma.ps1                         Gemma 4 26B daily driver + vision
      .\run-gemma.ps1 -Mtp                    Gemma 4 26B assistant MTP + vision
      .\run-gemma.ps1 -g12                    Gemma 4 12B dense + vision
      .\run-gemma.ps1 -Qwen                   Qwen 3.6 NextN, target turbo4/turbo3
      .\run-gemma.ps1 -Ternary                Bonsai 27B DSpark + vision
      .\run-gemma.ps1 -Ternary -TernaryPreset 64k
      .\run-gemma.ps1 -Ternary -TernaryPreset 128k

    Ternary variants:
      .\run-gemma.ps1 -Ternary -NoSpec        target-only + vision
      .\run-gemma.ps1 -Ternary -NoVision      DSpark text-only
      .\run-gemma.ps1 -Ternary -NoSpec -NoVision
      .\run-gemma.ps1 -Ternary -FullKV        full-f16 target; Q8_0 draft KV

    Use -Preview to print the resolved JSON command/environment without starting
    the server. Unknown native llama-server arguments may be appended directly and
    are preserved as literal argv entries; structured flags must use wrapper fields.
#>
[CmdletBinding()]
param(
    [int] $Port = 8081,
    [int] $CtxSize = -1,              # -1 keeps the selected profile default; 0 means trained context
    [int] $NCpuMoe = -1,              # -1 keeps Gemma/Qwen profile defaults
    [switch] $Qwen,
    [switch] $Ternary,
    [ValidateSet('standard', '32k', '64k', '128k')]
    [string] $TernaryPreset = 'standard',
    [switch] $g12,
    [switch] $Mtp,
    [switch] $NoSpec,
    [int] $DraftBlockSize = -1,       # -1 keeps the selected profile's draft depth
    [switch] $NoVision,
    [switch] $FullKV,
    [switch] $Silly,
    [switch] $NoPrefetch,
    [int] $PrefetchSlots = 1,         # 1 selects the runtime's measured three-slot default
    [switch] $PinHost,
    [Nullable[int]] $TurboAdaptive,
    [int] $UbatchSize = 0,
    [string] $ServerPath,
    [string] $ModelPath,
    [string] $DraftPath,
    [string] $MmprojPath,
    [switch] $Preview,
    [switch] $SkipFileValidation,
    [switch] $NoCapabilityProbe,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Extra
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'scripts\atomic-launcher.ps1'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Canonical Atomic launcher not found: $launcher"
}

$selectors = @($Qwen, $Ternary, $g12) | Where-Object { $_ }
if ($selectors.Count -gt 1) { throw '-Qwen, -Ternary, and -g12 are mutually exclusive.' }
if ($Mtp -and ($Qwen -or $Ternary -or $g12)) { throw '-Mtp is only available for Gemma 4 26B.' }
if ($Mtp -and $NoSpec) { throw '-Mtp and -NoSpec conflict.' }
if ($TernaryPreset -ne 'standard' -and -not $Ternary) { throw '-TernaryPreset requires -Ternary.' }
if ($Ternary -and $NCpuMoe -ge 0) { throw 'Ternary Bonsai is not an MoE model; -NCpuMoe does not apply.' }
if ($g12 -and $NCpuMoe -ge 0) { throw 'Gemma 4 12B is dense; -NCpuMoe does not apply.' }
if ($PrefetchSlots -lt 1) { throw '-PrefetchSlots must be at least 1.' }

$stack = if ($Ternary) { 'ternary' } elseif ($Qwen) { 'qwen' } elseif ($g12) { 'gemma12' } else { 'gemma26' }
$resolvedTernaryPreset = $TernaryPreset
if ($Ternary -and $TernaryPreset -eq 'standard') {
    # Existing `-CtxSize` commands automatically receive the validated tier
    # settings; `-TernaryPreset 32k` remains an explicit way to keep 32K tuning.
    if ($CtxSize -eq 65536) { $resolvedTernaryPreset = '64k' }
    if ($CtxSize -eq 131072) { $resolvedTernaryPreset = '128k' }
}
$preset = if ($stack -eq 'gemma26' -and $Mtp) {
    'mtp'
} elseif ($Ternary) {
    $resolvedTernaryPreset
} else {
    'standard'
}

$overrides = [ordered]@{ port = $Port }
if ($CtxSize -ge 0) { $overrides.ctx_size = $CtxSize }
if ($NCpuMoe -ge 0) { $overrides.n_cpu_moe = $NCpuMoe }
if ($DraftBlockSize -ge 1) { $overrides.spec_draft_n_max = $DraftBlockSize }
if ($UbatchSize -gt 0) {
    $overrides.batch_size = 2048
    $overrides.ubatch_size = $UbatchSize
}

if ($NoSpec) {
    $overrides.spec_enabled = $false
    $overrides.spec_type = 'none'
}
if ($NoVision) { $overrides.vision_enabled = $false }
if ($FullKV) {
    $overrides.cache_type_k = 'f16'
    $overrides.cache_type_v = 'f16'
    if ($stack -eq 'gemma26') {
        $overrides.draft_cache_type_k = 'f16'
        $overrides.draft_cache_type_v = 'f16'
    }
}
if ($Silly) { $overrides.reasoning = 'off' }
if ($NoPrefetch) { $overrides.prefetch_experts = $false }
if (-not $NoPrefetch) { $overrides.prefetch_slots = $PrefetchSlots }
if ($PinHost) {
    $overrides.pin_host = $true
    $overrides.no_mmap = $false
}
if ($null -ne $TurboAdaptive) { $overrides.turbo_layer_adaptive = [int] $TurboAdaptive }
if (-not [string]::IsNullOrWhiteSpace($ServerPath)) { $overrides.server = $ServerPath }
if (-not [string]::IsNullOrWhiteSpace($ModelPath)) { $overrides.model = $ModelPath }
if (-not [string]::IsNullOrWhiteSpace($DraftPath)) { $overrides.draft_model = $DraftPath }
if (-not [string]::IsNullOrWhiteSpace($MmprojPath)) { $overrides.mmproj = $MmprojPath }
if ($Extra) { $overrides.extra_arguments = @($Extra) }

$temporaryConfig = New-TemporaryFile
try {
    [System.IO.File]::WriteAllText($temporaryConfig.FullName, ($overrides | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
    $action = if ($Preview) { 'Preview' } else { 'Launch' }
    & $launcher -Action $action -Stack $stack -Preset $preset -ConfigPath $temporaryConfig.FullName `
        -SkipFileValidation:$SkipFileValidation -NoCapabilityProbe:$NoCapabilityProbe -Json:$Preview
    exit $LASTEXITCODE
} finally {
    Remove-Item -LiteralPath $temporaryConfig.FullName -Force -ErrorAction SilentlyContinue
}
