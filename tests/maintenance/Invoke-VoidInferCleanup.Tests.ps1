$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$tool = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\tools\maintenance\Invoke-VoidInferCleanup.ps1'))
$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) ('voidinfer-cleanup-test-' + [guid]::NewGuid().ToString('N'))
$allowed = Join-Path $fixtureRoot 'allowed'
$outside = Join-Path $fixtureRoot 'outside'
New-Item -ItemType Directory -Path $allowed, $outside | Out-Null

function New-InventoryEntry {
    param(
        [Parameter(Mandatory)] [string] $Path,
        [string] $Decision = 'delete_after_verification'
    )
    $inventory = & $tool -InventoryPath $Path | ConvertFrom-Json
    return [ordered]@{
        path = $inventory.path
        file_count = $inventory.file_count
        directory_count = $inventory.directory_count
        logical_bytes = $inventory.logical_bytes
        inventory_sha256 = $inventory.inventory_sha256
        content_sha256 = if (Test-Path -LiteralPath $Path -PathType Leaf) {
            (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
        } else { $null }
        owner_project = 'VoidInfer test fixture'
        reason = 'Disposable cleanup safety test'
        preservation_action = 'None required for disposable fixture'
        decision = $Decision
    }
}

function Write-Manifest {
    param(
        [Parameter(Mandatory)] [string] $Path,
        [Parameter(Mandatory)] [object[]] $Candidates,
        [string[]] $ProtectedPaths = @()
    )
    $body = [ordered]@{
        schema_version = 1
        allowed_roots = @($allowed)
        protected_paths = @($ProtectedPaths)
        candidates = @($Candidates)
    } | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText($Path, $body, [Text.UTF8Encoding]::new($false))
}

function Assert-Throws {
    param([Parameter(Mandatory)] [scriptblock] $Action, [Parameter(Mandatory)] [string] $Label)
    $threw = $false
    try { & $Action } catch { $threw = $true }
    if (-not $threw) { throw "Expected failure was not raised: $Label" }
}

try {
    $dry = Join-Path $allowed 'dry-run'
    New-Item -ItemType Directory -Path $dry | Out-Null
    [IO.File]::WriteAllText((Join-Path $dry 'payload.txt'), 'dry-run payload')
    $manifest = Join-Path $fixtureRoot 'dry-run.json'
    Write-Manifest -Path $manifest -Candidates @(New-InventoryEntry -Path $dry)
    & $tool -Manifest $manifest -LogPath (Join-Path $fixtureRoot 'dry-run.jsonl') | Out-Null
    if (-not (Test-Path -LiteralPath $dry)) { throw 'Dry-run deleted its candidate.' }

    $apply = Join-Path $allowed 'apply'
    New-Item -ItemType Directory -Path $apply | Out-Null
    [IO.File]::WriteAllText((Join-Path $apply 'payload.txt'), 'apply payload')
    $manifest = Join-Path $fixtureRoot 'apply.json'
    Write-Manifest -Path $manifest -Candidates @(New-InventoryEntry -Path $apply)
    & $tool -Manifest $manifest -Apply -LogPath (Join-Path $fixtureRoot 'apply.jsonl') | Out-Null
    if (Test-Path -LiteralPath $apply) { throw 'Apply mode did not delete its validated candidate.' }

    $protected = Join-Path $allowed 'protected'
    New-Item -ItemType Directory -Path $protected | Out-Null
    [IO.File]::WriteAllText((Join-Path $protected 'keep.txt'), 'protected')
    $manifest = Join-Path $fixtureRoot 'protected.json'
    Write-Manifest -Path $manifest -Candidates @(New-InventoryEntry -Path $protected) -ProtectedPaths @($protected)
    Assert-Throws -Label 'protected candidate' -Action {
        & $tool -Manifest $manifest -LogPath (Join-Path $fixtureRoot 'protected.jsonl') | Out-Null
    }
    if (-not (Test-Path -LiteralPath $protected)) { throw 'Protected fixture was deleted.' }

    $changed = Join-Path $allowed 'changed'
    New-Item -ItemType Directory -Path $changed | Out-Null
    $changedFile = Join-Path $changed 'payload.txt'
    [IO.File]::WriteAllText($changedFile, 'before')
    $manifest = Join-Path $fixtureRoot 'changed.json'
    Write-Manifest -Path $manifest -Candidates @(New-InventoryEntry -Path $changed)
    [IO.File]::WriteAllText($changedFile, 'after with a different length')
    Assert-Throws -Label 'changed candidate' -Action {
        & $tool -Manifest $manifest -Apply -LogPath (Join-Path $fixtureRoot 'changed.jsonl') | Out-Null
    }
    if (-not (Test-Path -LiteralPath $changed)) { throw 'Changed fixture was deleted.' }

    $outsideCandidate = Join-Path $outside 'candidate'
    New-Item -ItemType Directory -Path $outsideCandidate | Out-Null
    [IO.File]::WriteAllText((Join-Path $outsideCandidate 'payload.txt'), 'outside')
    $manifest = Join-Path $fixtureRoot 'outside.json'
    Write-Manifest -Path $manifest -Candidates @(New-InventoryEntry -Path $outsideCandidate)
    Assert-Throws -Label 'outside allowlist' -Action {
        & $tool -Manifest $manifest -LogPath (Join-Path $fixtureRoot 'outside.jsonl') | Out-Null
    }

    $nested = Join-Path $allowed 'nested-git'
    New-Item -ItemType Directory -Path (Join-Path $nested '.git') -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $nested '.git\config'), '[core]')
    Assert-Throws -Label 'nested repository' -Action {
        & $tool -InventoryPath $nested | Out-Null
    }

    $junctionTarget = Join-Path $outside 'junction-target'
    $junctionCandidate = Join-Path $allowed 'junction-candidate'
    New-Item -ItemType Directory -Path $junctionTarget, $junctionCandidate | Out-Null
    [IO.File]::WriteAllText((Join-Path $junctionTarget 'keep.txt'), 'junction target')
    New-Item -ItemType Junction -Path (Join-Path $junctionCandidate 'escape') -Target $junctionTarget | Out-Null
    Assert-Throws -Label 'junction escape' -Action {
        & $tool -InventoryPath $junctionCandidate | Out-Null
    }
    if (-not (Test-Path -LiteralPath (Join-Path $junctionTarget 'keep.txt'))) {
        throw 'Junction target was modified.'
    }

    Write-Output 'PASS: cleanup utility safety fixtures'
} finally {
    if (Test-Path -LiteralPath $fixtureRoot) {
        Remove-Item -LiteralPath $fixtureRoot -Recurse -Force
    }
}
