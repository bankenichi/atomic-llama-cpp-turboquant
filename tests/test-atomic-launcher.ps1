[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$modulePath = Join-Path $repoRoot 'scripts\atomic-launcher\AtomicLauncher.psm1'
$profilesPath = Join-Path $repoRoot 'scripts\atomic-launcher\profiles.json'

Import-Module $modulePath -Force

$script:failures = [System.Collections.Generic.List[string]]::new()
$script:passes = 0

function Assert-Equal {
    param([object] $Actual, [object] $Expected, [string] $Message)
    if ($Actual -ne $Expected) {
        $script:failures.Add("$Message (expected '$Expected', got '$Actual')")
    } else {
        $script:passes++
    }
}

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) {
        $script:failures.Add($Message)
    } else {
        $script:passes++
    }
}

function Assert-Contains {
    param([object[]] $Collection, [object] $Value, [string] $Message)
    Assert-True -Condition ($Collection -contains $Value) -Message $Message
}

function Get-ArgumentValue {
    param([object[]] $Arguments, [string] $Name)
    $index = [Array]::IndexOf($Arguments, $Name)
    if ($index -lt 0 -or $index + 1 -ge $Arguments.Count) { return $null }
    return $Arguments[$index + 1]
}

$catalog = Get-AtomicProfileCatalog -ProfilesPath $profilesPath
Assert-Equal $catalog.schema_version 1 'Profile schema version is stable'
Assert-Contains @($catalog.stacks.PSObject.Properties.Name) 'ternary' 'Ternary stack is present'

$qwen = Resolve-AtomicConfiguration -ProfilesPath $profilesPath -Stack qwen -Preset standard
Assert-Equal $qwen.cache_type_k 'turbo4' 'Qwen target K cache uses the owner standard'
Assert-Equal $qwen.cache_type_v 'turbo3' 'Qwen target V cache uses the owner standard'
Assert-Equal $qwen.draft_cache_type_k 'q8_0' 'Qwen draft K cache uses lossless Q8_0'
Assert-Equal $qwen.draft_cache_type_v 'q8_0' 'Qwen draft V cache uses lossless Q8_0'
Assert-Equal $qwen.spec_type 'draft-mtp' 'Qwen uses the current embedded NextN CLI type'
Assert-True ([string]::IsNullOrWhiteSpace([string] $qwen.draft_model)) 'Qwen does not map a second GGUF'

$qwenPlan = New-AtomicLaunchPlan -Configuration $qwen -SkipFileValidation
Assert-True $qwenPlan.validation.valid 'Standard Qwen plan validates'
Assert-Equal (Get-ArgumentValue $qwenPlan.arguments '--cache-type-k') 'turbo4' 'Qwen command emits target K cache'
Assert-Equal (Get-ArgumentValue $qwenPlan.arguments '--cache-type-v') 'turbo3' 'Qwen command emits target V cache'
Assert-Equal (Get-ArgumentValue $qwenPlan.arguments '--spec-draft-type-k') 'q8_0' 'Qwen command emits lossless draft K cache'
Assert-Equal (Get-ArgumentValue $qwenPlan.arguments '--spec-draft-type-v') 'q8_0' 'Qwen command emits lossless draft V cache'
Assert-Equal (Get-ArgumentValue $qwenPlan.arguments '--load-mode') 'mlock' 'Launcher emits the unambiguous modern load mode'
Assert-True (-not ($qwenPlan.arguments -contains '--mlock')) 'Launcher omits deprecated --mlock'
Assert-True (-not ($qwenPlan.arguments -contains '--no-mmap')) 'Launcher omits deprecated --no-mmap'
Assert-Equal $qwenPlan.environment.GGML_SCHED_PREFETCH_EXPERTS '1' 'Qwen enables expert prefetch'

$ternary = Resolve-AtomicConfiguration -ProfilesPath $profilesPath -Stack ternary -Preset standard
Assert-Equal $ternary.spec_type 'draft-dspark' 'Ternary standard uses DSpark'
Assert-Equal $ternary.spec_draft_n_max 4 'Ternary standard uses the tested four-token draft'
Assert-Equal $ternary.cache_type_k 'turbo4' 'Ternary target K cache uses the owner standard'
Assert-Equal $ternary.cache_type_v 'turbo3' 'Ternary target V cache uses the owner standard'
Assert-Equal $ternary.draft_cache_type_k 'q8_0' 'Ternary draft K cache uses lossless Q8_0'
Assert-Equal $ternary.draft_cache_type_v 'q8_0' 'Ternary draft V cache uses lossless Q8_0'
Assert-True $ternary.vision_enabled 'Ternary standard keeps vision enabled'
Assert-True $ternary.model.EndsWith('Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf') 'Ternary target path matches the local artifact'
Assert-True $ternary.draft_model.EndsWith('Ternary-Bonsai-27B-dspark-Q4_1.gguf') 'Ternary draft path matches the local artifact'
Assert-True $ternary.mmproj.EndsWith('Ternary-Bonsai-27B-mmproj-Q8_0.gguf') 'Ternary mmproj path matches the local artifact'

$ternaryPlan = New-AtomicLaunchPlan -Configuration $ternary -SkipFileValidation
Assert-True $ternaryPlan.validation.valid 'Standard Ternary plan validates'
Assert-Equal (Get-ArgumentValue $ternaryPlan.arguments '--spec-type') 'draft-dspark' 'Ternary command emits DSpark'
Assert-Equal (Get-ArgumentValue $ternaryPlan.arguments '--cache-type-k') 'turbo4' 'Ternary command emits target K cache'
Assert-Equal (Get-ArgumentValue $ternaryPlan.arguments '--cache-type-v') 'turbo3' 'Ternary command emits target V cache'
Assert-Equal (Get-ArgumentValue $ternaryPlan.arguments '--spec-draft-type-k') 'q8_0' 'Ternary command emits lossless draft K cache'
Assert-Equal (Get-ArgumentValue $ternaryPlan.arguments '--spec-draft-type-v') 'q8_0' 'Ternary command emits lossless draft V cache'
Assert-Equal (Get-ArgumentValue $ternaryPlan.arguments '--model-draft') $ternary.draft_model 'Ternary command emits the sidecar path literally'
Assert-True (-not ($ternaryPlan.arguments -contains '--spec-draft-ctx-size')) 'Launcher never emits the stale draft-context flag'

$ternaryFull = Resolve-AtomicConfiguration -ProfilesPath $profilesPath -Stack ternary -Preset full-f16
Assert-Equal $ternaryFull.cache_type_k 'f16' 'Full-f16 Ternary target K cache'
Assert-Equal $ternaryFull.cache_type_v 'f16' 'Full-f16 Ternary target V cache'
Assert-Equal $ternaryFull.draft_cache_type_k 'q8_0' 'Target-f16 diagnostic keeps lossless Q8_0 draft K cache'
Assert-Equal $ternaryFull.draft_cache_type_v 'q8_0' 'Target-f16 diagnostic keeps lossless Q8_0 draft V cache'

$gemma = Resolve-AtomicConfiguration -ProfilesPath $profilesPath -Stack gemma26 -Preset mtp
Assert-Equal $gemma.spec_type 'draft-mtp' 'Gemma MTP uses the current CLI type'
Assert-True $gemma.draft_model.EndsWith('mtp-gemma-4-26B-A4B-it.gguf') 'Gemma assistant path uses Program Files artifact'
Assert-Equal $gemma.draft_cache_type_k 'turbo3' 'Gemma assistant K cache is preserved'
Assert-Equal $gemma.draft_cache_type_v 'turbo3' 'Gemma assistant V cache is preserved'

$override = Resolve-AtomicConfiguration -ProfilesPath $profilesPath -Stack ternary -Preset standard -Overrides @{
    vision_enabled = $false
    spec_enabled = $false
    port = 9000
}
$overridePlan = New-AtomicLaunchPlan -Configuration $override -SkipFileValidation
Assert-True (-not ($overridePlan.arguments -contains '--mmproj')) 'NoVision removes mmproj'
Assert-True (-not ($overridePlan.arguments -contains '--spec-type')) 'NoSpec removes speculative arguments'
Assert-Equal (Get-ArgumentValue $overridePlan.arguments '--port') 9000 'Explicit override wins over profile'

$badDraft = Resolve-AtomicConfiguration -ProfilesPath $profilesPath -Stack ternary -Preset standard -Overrides @{
    spec_draft_n_max = 17
}
$badDraftResult = Test-AtomicConfiguration -Configuration $badDraft -SkipFileValidation
Assert-True (-not $badDraftResult.valid) 'DSpark rejects more than 16 draft slots'
Assert-True (($badDraftResult.errors -join "`n") -match '16') 'DSpark slot error explains the bound'

$missingDraft = Resolve-AtomicConfiguration -ProfilesPath $profilesPath -Stack ternary -Preset standard -Overrides @{
    draft_model = ''
}
$missingDraftResult = Test-AtomicConfiguration -Configuration $missingDraft -SkipFileValidation
Assert-True (-not $missingDraftResult.valid) 'DSpark requires a sidecar model'

$helpSample = @'
--spec-type none,draft-mtp,draft-dspark
--spec-draft-n-max N
--cache-type-k TYPE
--cache-type-v TYPE
--spec-draft-type-k TYPE
--spec-draft-type-v TYPE
--mmproj FILE
--context-shift
--metrics
'@
$capabilities = Get-AtomicServerCapabilities -HelpText $helpSample
Assert-Contains $capabilities.spec_types 'draft-dspark' 'Capability parser discovers DSpark'
Assert-Contains $capabilities.spec_types 'draft-mtp' 'Capability parser discovers MTP'
Assert-True $capabilities.flags.'--metrics' 'Capability parser discovers metrics'

$rawConflict = Resolve-AtomicConfiguration -ProfilesPath $profilesPath -Stack qwen -Preset standard -Overrides @{
    extra_arguments = @('--cache-type-k', 'f16')
}
$rawConflictResult = Test-AtomicConfiguration -Configuration $rawConflict -SkipFileValidation
Assert-True (-not $rawConflictResult.valid) 'Raw arguments cannot override structured settings silently'

if ($script:failures.Count -gt 0) {
    Write-Error ("Atomic launcher tests failed ({0} passed, {1} failed):`n - {2}" -f $script:passes, $script:failures.Count, ($script:failures -join "`n - "))
}

Write-Host "Atomic launcher tests passed: $script:passes" -ForegroundColor Green
