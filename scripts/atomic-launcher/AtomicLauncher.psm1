Set-StrictMode -Version Latest

$script:RepositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$script:ProfilesPath = Join-Path $PSScriptRoot 'profiles.json'

function ConvertTo-AtomicMap {
    param([Parameter(ValueFromPipeline = $true)] [object] $InputObject)

    if ($null -eq $InputObject) { return $null }
    # PowerShell's pipeline adapter can present strings as PSCustomObject values
    # exposing a synthetic Length property. Keep scalar values ahead of object
    # traversal so argument arrays remain arrays of literal strings.
    if ($InputObject -is [string] -or $InputObject.GetType().IsValueType) {
        return $InputObject
    }
    if ($InputObject -is [System.Collections.IDictionary]) {
        $map = [ordered]@{}
        foreach ($key in $InputObject.Keys) {
            $map[[string] $key] = ConvertTo-AtomicMap $InputObject[$key]
        }
        return $map
    }
    if ($InputObject -is [pscustomobject]) {
        $map = [ordered]@{}
        foreach ($property in $InputObject.PSObject.Properties) {
            $map[$property.Name] = ConvertTo-AtomicMap $property.Value
        }
        return $map
    }
    if ($InputObject -is [System.Collections.IEnumerable] -and $InputObject -isnot [string]) {
        return @($InputObject | ForEach-Object { ConvertTo-AtomicMap $_ })
    }
    return $InputObject
}

function Merge-AtomicMap {
    param(
        [Parameter(Mandatory = $true)] [System.Collections.IDictionary] $Target,
        [Parameter(Mandatory = $true)] [System.Collections.IDictionary] $Source
    )

    foreach ($key in $Source.Keys) {
        $sourceValue = $Source[$key]
        if ($sourceValue -is [System.Collections.IDictionary] -and
            $Target.Contains($key) -and
            $Target[$key] -is [System.Collections.IDictionary]) {
            Merge-AtomicMap -Target $Target[$key] -Source $sourceValue
        } else {
            $Target[$key] = $sourceValue
        }
    }
}

function Get-AtomicProfileCatalog {
    [CmdletBinding()]
    param([string] $ProfilesPath = $script:ProfilesPath)

    if (-not (Test-Path -LiteralPath $ProfilesPath -PathType Leaf)) {
        throw "Atomic launcher profile catalog not found: $ProfilesPath"
    }
    return Get-Content -LiteralPath $ProfilesPath -Raw | ConvertFrom-Json
}

function Get-AtomicDefaultServerPath {
    $candidates = @(
        $env:ATOMIC_LLAMA_SERVER,
        (Join-Path $script:RepositoryRoot 'build-convergence\bin\Release\llama-server.exe'),
        (Join-Path $script:RepositoryRoot 'build\bin\Release\llama-server.exe'),
        'C:\Program Files\llamacpp\llama-server.exe'
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    return [string] $candidates[0]
}

function Resolve-AtomicConfiguration {
    [CmdletBinding()]
    param(
        [string] $ProfilesPath = $script:ProfilesPath,
        [Parameter(Mandatory = $true)] [string] $Stack,
        [string] $Preset = 'standard',
        [System.Collections.IDictionary] $Overrides = @{}
    )

    $catalog = Get-AtomicProfileCatalog -ProfilesPath $ProfilesPath
    $stackProperty = $catalog.stacks.PSObject.Properties[$Stack]
    if ($null -eq $stackProperty) {
        $known = $catalog.stacks.PSObject.Properties.Name -join ', '
        throw "Unknown Atomic stack '$Stack'. Available stacks: $known"
    }

    $stackDefinition = $stackProperty.Value
    $presetProperty = $stackDefinition.presets.PSObject.Properties[$Preset]
    if ($null -eq $presetProperty) {
        $known = $stackDefinition.presets.PSObject.Properties.Name -join ', '
        throw "Unknown preset '$Preset' for stack '$Stack'. Available presets: $known"
    }

    $settings = ConvertTo-AtomicMap $catalog.defaults
    Merge-AtomicMap -Target $settings -Source (ConvertTo-AtomicMap $stackDefinition.base)
    Merge-AtomicMap -Target $settings -Source (ConvertTo-AtomicMap $presetProperty.Value)
    if ($Overrides.Count -gt 0) {
        Merge-AtomicMap -Target $settings -Source (ConvertTo-AtomicMap $Overrides)
    }

    $settings['schema_version'] = [int] $catalog.schema_version
    $settings['stack'] = $Stack
    $settings['stack_label'] = [string] $stackDefinition.label
    $settings['preset'] = $Preset

    if (-not $settings.Contains('server') -or [string]::IsNullOrWhiteSpace([string] $settings.server)) {
        $settings['server'] = Get-AtomicDefaultServerPath
    }

    $modelRoot = if ($settings.Contains('model_root') -and -not [string]::IsNullOrWhiteSpace([string] $settings.model_root)) {
        [string] $settings.model_root
    } elseif (-not [string]::IsNullOrWhiteSpace($env:ATOMIC_MODEL_ROOT)) {
        [string] $env:ATOMIC_MODEL_ROOT
    } else {
        'C:\Program Files\llamacpp'
    }
    $settings['model_root'] = $modelRoot

    foreach ($pair in @(
        @('model', 'model_file'),
        @('draft_model', 'draft_model_file'),
        @('mmproj', 'mmproj_file')
    )) {
        $pathKey = $pair[0]
        $fileKey = $pair[1]
        if (-not $Overrides.Contains($pathKey) -and
            (!$settings.Contains($pathKey) -or [string]::IsNullOrWhiteSpace([string] $settings[$pathKey])) -and
            $settings.Contains($fileKey) -and
            -not [string]::IsNullOrWhiteSpace([string] $settings[$fileKey])) {
            $settings[$pathKey] = Join-Path $modelRoot ([string] $settings[$fileKey])
        }
    }

    if ($settings.extra_arguments -is [string]) {
        $settings.extra_arguments = @([string] $settings.extra_arguments)
    } elseif ($null -eq $settings.extra_arguments) {
        $settings.extra_arguments = @()
    } else {
        $settings.extra_arguments = @($settings.extra_arguments)
    }

    return [pscustomobject] $settings
}

function Get-AtomicServerCapabilities {
    [CmdletBinding(DefaultParameterSetName = 'Path')]
    param(
        [Parameter(ParameterSetName = 'Path', Mandatory = $true)] [string] $ServerPath,
        [Parameter(ParameterSetName = 'Text', Mandatory = $true)] [string] $HelpText
    )

    if ($PSCmdlet.ParameterSetName -eq 'Path') {
        if (-not (Test-Path -LiteralPath $ServerPath -PathType Leaf)) {
            throw "llama-server executable not found: $ServerPath"
        }
        $HelpText = (& $ServerPath --help 2>&1 | Out-String)
    }

    $flags = [ordered]@{}
    foreach ($match in [regex]::Matches($HelpText, '--[a-z0-9][a-z0-9-]*', 'IgnoreCase')) {
        $flags[$match.Value.ToLowerInvariant()] = $true
    }

    $specTypes = @()
    $specMatch = [regex]::Match($HelpText, '--spec-type\s+([^\r\n]+)', 'IgnoreCase')
    if ($specMatch.Success) {
        $typeToken = ($specMatch.Groups[1].Value.Trim() -split '\s+')[0]
        $specTypes = @($typeToken -split ',' | Where-Object { $_ })
    }

    return [pscustomobject] [ordered]@{
        flags = [pscustomobject] $flags
        spec_types = $specTypes
        cache_types = @('f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1', 'turbo2', 'turbo3', 'turbo4')
        help_text = $HelpText
    }
}

function Test-AtomicConfiguration {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)] [psobject] $Configuration,
        [switch] $SkipFileValidation,
        [psobject] $Capabilities
    )

    $errors = [System.Collections.Generic.List[string]]::new()
    $warnings = [System.Collections.Generic.List[string]]::new()

    if ([string]::IsNullOrWhiteSpace([string] $Configuration.server)) {
        $errors.Add('A llama-server executable path is required.')
    } elseif (-not $SkipFileValidation -and -not (Test-Path -LiteralPath $Configuration.server -PathType Leaf)) {
        $errors.Add("llama-server executable not found: $($Configuration.server)")
    }

    if ([string]::IsNullOrWhiteSpace([string] $Configuration.model)) {
        $errors.Add('A target model path is required.')
    } elseif (-not $SkipFileValidation -and -not (Test-Path -LiteralPath $Configuration.model -PathType Leaf)) {
        $errors.Add("Target model not found: $($Configuration.model)")
    }

    if ([int] $Configuration.port -lt 1 -or [int] $Configuration.port -gt 65535) {
        $errors.Add("Port must be between 1 and 65535; got $($Configuration.port).")
    }
    if ([int64] $Configuration.ctx_size -lt 0) {
        $errors.Add("Context size cannot be negative; got $($Configuration.ctx_size).")
    }
    if ([int] $Configuration.n_cpu_moe -lt 0) {
        $errors.Add("CPU MoE layer count cannot be negative; got $($Configuration.n_cpu_moe).")
    }

    if ([bool] $Configuration.vision_enabled) {
        if ([string]::IsNullOrWhiteSpace([string] $Configuration.mmproj)) {
            $errors.Add('Vision is enabled but no mmproj path is configured.')
        } elseif (-not $SkipFileValidation -and -not (Test-Path -LiteralPath $Configuration.mmproj -PathType Leaf)) {
            $errors.Add("mmproj not found: $($Configuration.mmproj)")
        }
    }

    if ([bool] $Configuration.spec_enabled) {
        if ([string]::IsNullOrWhiteSpace([string] $Configuration.spec_type) -or $Configuration.spec_type -eq 'none') {
            $errors.Add('Speculative decoding is enabled but spec_type is none.')
        }
        if ([int] $Configuration.spec_draft_n_max -lt 1) {
            $errors.Add('Speculative draft maximum must be at least one token.')
        }

        switch ([string] $Configuration.stack) {
            'ternary' {
                if ($Configuration.spec_type -ne 'draft-dspark') {
                    $errors.Add("Ternary Bonsai requires spec_type 'draft-dspark'; got '$($Configuration.spec_type)'.")
                }
                if ([string]::IsNullOrWhiteSpace([string] $Configuration.draft_model)) {
                    $errors.Add('Ternary DSpark requires the matching draft sidecar model.')
                } elseif (-not $SkipFileValidation -and -not (Test-Path -LiteralPath $Configuration.draft_model -PathType Leaf)) {
                    $errors.Add("DSpark draft model not found: $($Configuration.draft_model)")
                }
                $slotLimit = if ($Configuration.PSObject.Properties['draft_slot_limit']) { [int] $Configuration.draft_slot_limit } else { 16 }
                if ([int] $Configuration.spec_draft_n_max -gt $slotLimit) {
                    $errors.Add("Ternary DSpark supports at most $slotLimit draft slots; got $($Configuration.spec_draft_n_max).")
                }
            }
            'qwen' {
                if ($Configuration.spec_type -ne 'draft-mtp') {
                    $errors.Add("Qwen embedded NextN requires spec_type 'draft-mtp'; got '$($Configuration.spec_type)'.")
                }
                if (-not [string]::IsNullOrWhiteSpace([string] $Configuration.draft_model)) {
                    $errors.Add('Qwen embedded NextN reuses the target GGUF; draft_model must be empty.')
                }
            }
            'gemma26' {
                if ($Configuration.spec_type -ne 'draft-mtp') {
                    $errors.Add("Gemma assistant decoding requires spec_type 'draft-mtp'; got '$($Configuration.spec_type)'.")
                }
                if ([string]::IsNullOrWhiteSpace([string] $Configuration.draft_model)) {
                    $errors.Add('Gemma MTP requires an assistant GGUF.')
                } elseif (-not $SkipFileValidation -and -not (Test-Path -LiteralPath $Configuration.draft_model -PathType Leaf)) {
                    $errors.Add("Gemma assistant model not found: $($Configuration.draft_model)")
                }
            }
            'gemma12' {
                $errors.Add('Gemma 4 12B has no configured speculative assistant.')
            }
        }
    }

    $structuredFlags = @(
        '--model', '-m', '--ctx-size', '-c', '--n-gpu-layers', '-ngl', '--gpu-layers',
        '--keep', '--parallel', '-np', '--host', '--port', '--cache-type-k', '-ctk',
        '--cache-type-v', '-ctv', '--mmproj', '-mm', '--spec-type', '--model-draft',
        '--spec-draft-model', '-md', '--spec-draft-n-max', '--spec-draft-type-k', '-ctkd',
        '--spec-draft-type-v', '-ctvd', '--spec-draft-ngl', '-ngld', '--n-cpu-moe',
        '-ncmoe', '--batch-size', '-b', '--ubatch-size', '-ub', '--flash-attn', '-fa',
        '--reasoning', '-rea', '--context-shift', '--no-context-shift'
    )
    foreach ($token in @($Configuration.extra_arguments)) {
        if ([string] $token -in $structuredFlags) {
            $errors.Add("Raw argument '$token' conflicts with a structured Atomic setting. Use the structured field instead.")
        }
    }

    if ($null -ne $Capabilities) {
        $requiredFlags = @('--model', '--ctx-size', '--cache-type-k', '--cache-type-v')
        if ([bool] $Configuration.vision_enabled) { $requiredFlags += '--mmproj' }
        if ([bool] $Configuration.spec_enabled) {
            $requiredFlags += @('--spec-type', '--spec-draft-n-max', '--spec-draft-type-k', '--spec-draft-type-v')
            if (-not [string]::IsNullOrWhiteSpace([string] $Configuration.draft_model)) { $requiredFlags += '--model-draft' }
        }
        foreach ($flag in $requiredFlags) {
            if (-not $Capabilities.flags.PSObject.Properties[$flag]) {
                $errors.Add("Selected llama-server does not advertise required option $flag.")
            }
        }
        if ([bool] $Configuration.spec_enabled -and $Capabilities.spec_types -notcontains $Configuration.spec_type) {
            $errors.Add("Selected llama-server does not advertise speculative type '$($Configuration.spec_type)'.")
        }
    }

    if ([bool] $Configuration.context_shift -and ([string] $Configuration.cache_type_k).StartsWith('turbo')) {
        $warnings.Add('TurboQuant K cannot shift in place; Atomic will use its transactional reprefill path.')
    }
    if ([bool] $Configuration.vision_enabled -and [bool] $Configuration.spec_enabled) {
        $warnings.Add('Real media uses the selected speculative implementation policy. The target still verifies every proposal, and the configured draft depth remains in effect.')
    }
    if ($Configuration.cache_type_k -eq 'f16' -and $Configuration.cache_type_v -eq 'f16') {
        $warnings.Add('Full-f16 target KV consumes substantially more VRAM than the tuned cache profile.')
    }
    if ([bool] $Configuration.prefetch_experts -and [bool] $Configuration.pin_host) {
        $warnings.Add('Host registration negates the measured expert-prefetch benefit; use one path at a time for clean comparisons.')
    }

    return [pscustomobject] [ordered]@{
        valid = ($errors.Count -eq 0)
        errors = @($errors)
        warnings = @($warnings)
    }
}

function Add-AtomicArgument {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [System.Collections.Generic.List[object]] $List,
        [Parameter(Mandatory = $true)] [string] $Name,
        [object] $Value,
        [switch] $Switch
    )
    [void] $List.Add($Name)
    if (-not $Switch) { [void] $List.Add($Value) }
}

function Get-AtomicRedactedArguments {
    param([object[]] $Arguments)
    $secretFlags = @('--api-key', '--hf-token')
    $redacted = @($Arguments)
    for ($index = 0; $index -lt $redacted.Count - 1; $index++) {
        if ([string] $redacted[$index] -in $secretFlags) {
            $redacted[$index + 1] = '<redacted>'
            $index++
        }
    }
    return $redacted
}

function New-AtomicLaunchPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)] [psobject] $Configuration,
        [switch] $SkipFileValidation,
        [psobject] $Capabilities
    )

    $validation = Test-AtomicConfiguration -Configuration $Configuration -SkipFileValidation:$SkipFileValidation -Capabilities $Capabilities
    $arguments = [System.Collections.Generic.List[object]]::new()

    Add-AtomicArgument $arguments '--model' $Configuration.model
    Add-AtomicArgument $arguments '--ctx-size' ([int64] $Configuration.ctx_size)
    Add-AtomicArgument $arguments '--n-gpu-layers' ([int] $Configuration.gpu_layers)
    Add-AtomicArgument $arguments '--keep' ([int] $Configuration.keep)
    Add-AtomicArgument $arguments '--parallel' ([int] $Configuration.parallel)
    Add-AtomicArgument $arguments '--host' $Configuration.host
    Add-AtomicArgument $arguments '--port' ([int] $Configuration.port)
    Add-AtomicArgument $arguments '--cache-type-k' $Configuration.cache_type_k
    Add-AtomicArgument $arguments '--cache-type-v' $Configuration.cache_type_v

    # The deprecated switches both overwrite one load_mode field, so emitting
    # `--mlock --no-mmap` silently ended in mode `none`. Express the intended
    # non-mmap locked load atomically through the modern option.
    if ([bool] $Configuration.mlock) {
        Add-AtomicArgument $arguments '--load-mode' 'mlock'
    } elseif ([bool] $Configuration.no_mmap -and -not [bool] $Configuration.pin_host) {
        Add-AtomicArgument $arguments '--load-mode' 'none'
    }
    if ([bool] $Configuration.jinja) { Add-AtomicArgument $arguments '--jinja' -Switch }
    Add-AtomicArgument $arguments '--flash-attn' $Configuration.flash_attention
    Add-AtomicArgument $arguments $(if ([bool] $Configuration.context_shift) { '--context-shift' } else { '--no-context-shift' }) -Switch

    if ([bool] $Configuration.has_moe) {
        Add-AtomicArgument $arguments '--n-cpu-moe' ([int] $Configuration.n_cpu_moe)
    }
    if ([int] $Configuration.batch_size -gt 0) { Add-AtomicArgument $arguments '--batch-size' ([int] $Configuration.batch_size) }
    if ([int] $Configuration.ubatch_size -gt 0) { Add-AtomicArgument $arguments '--ubatch-size' ([int] $Configuration.ubatch_size) }
    if ([bool] $Configuration.vision_enabled) { Add-AtomicArgument $arguments '--mmproj' $Configuration.mmproj }

    if ([bool] $Configuration.spec_enabled) {
        Add-AtomicArgument $arguments '--spec-type' $Configuration.spec_type
        if (-not [string]::IsNullOrWhiteSpace([string] $Configuration.draft_model)) {
            Add-AtomicArgument $arguments '--model-draft' $Configuration.draft_model
        }
        Add-AtomicArgument $arguments '--spec-draft-ngl' ([int] $Configuration.draft_gpu_layers)
        Add-AtomicArgument $arguments '--spec-draft-type-k' $Configuration.draft_cache_type_k
        Add-AtomicArgument $arguments '--spec-draft-type-v' $Configuration.draft_cache_type_v
        Add-AtomicArgument $arguments '--spec-draft-n-max' ([int] $Configuration.spec_draft_n_max)
        if ([int] $Configuration.spec_draft_n_min -gt 0) {
            Add-AtomicArgument $arguments '--spec-draft-n-min' ([int] $Configuration.spec_draft_n_min)
        }
        if ([double] $Configuration.spec_draft_p_min -gt 0) {
            Add-AtomicArgument $arguments '--spec-draft-p-min' ([double] $Configuration.spec_draft_p_min)
        }
    }

    if ([string] $Configuration.reasoning -ne 'auto') { Add-AtomicArgument $arguments '--reasoning' $Configuration.reasoning }
    if ([bool] $Configuration.metrics) { Add-AtomicArgument $arguments '--metrics' -Switch }
    if (-not [bool] $Configuration.slots_endpoint) { Add-AtomicArgument $arguments '--no-slots' -Switch }

    foreach ($extra in @($Configuration.extra_arguments)) { [void] $arguments.Add($extra) }

    $environment = [ordered]@{}
    if ([bool] $Configuration.prefetch_experts -and [bool] $Configuration.has_moe) {
        $environment['GGML_SCHED_PREFETCH_EXPERTS'] = [string] $Configuration.prefetch_slots
    }
    if ([bool] $Configuration.pin_host) { $environment['GGML_CUDA_REGISTER_HOST'] = '1' }
    if ($null -ne $Configuration.turbo_layer_adaptive -and [string] $Configuration.turbo_layer_adaptive -ne '') {
        $environment['TURBO_LAYER_ADAPTIVE'] = [string] $Configuration.turbo_layer_adaptive
    }

    $argumentArray = $arguments.ToArray()
    return [pscustomobject] [ordered]@{
        schema_version = 1
        executable = [string] $Configuration.server
        arguments = $argumentArray
        redacted_arguments = @(Get-AtomicRedactedArguments -Arguments $argumentArray)
        environment = [pscustomobject] $environment
        validation = $validation
        configuration = $Configuration
    }
}

Export-ModuleMember -Function @(
    'Get-AtomicProfileCatalog',
    'Get-AtomicDefaultServerPath',
    'Resolve-AtomicConfiguration',
    'Get-AtomicServerCapabilities',
    'Test-AtomicConfiguration',
    'New-AtomicLaunchPlan'
)
