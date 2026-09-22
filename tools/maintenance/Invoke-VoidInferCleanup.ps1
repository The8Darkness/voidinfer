[CmdletBinding(DefaultParameterSetName = 'Run')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Run')]
    [string] $Manifest,

    [Parameter(ParameterSetName = 'Run')]
    [switch] $Apply,

    [Parameter(ParameterSetName = 'Run')]
    [string] $LogPath,

    [Parameter(Mandatory, ParameterSetName = 'Inventory')]
    [string] $InventoryPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-CanonicalExistingPath {
    param([Parameter(Mandatory)] [string] $Path)

    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction Stop
    return [IO.Path]::GetFullPath($resolved.Path).TrimEnd('\', '/')
}

function Test-PathWithin {
    param(
        [Parameter(Mandatory)] [string] $Path,
        [Parameter(Mandatory)] [string] $Root,
        [switch] $AllowEqual
    )

    if ($AllowEqual -and $Path.Equals($Root, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $prefix = $Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    return $Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Get-Inventory {
    param([Parameter(Mandatory)] [string] $Path)

    $canonical = Get-CanonicalExistingPath -Path $Path
    $rootItem = Get-Item -LiteralPath $canonical -Force
    if ($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Refusing reparse-point candidate root: $canonical"
    }

    $children = @(Get-ChildItem -LiteralPath $canonical -Force -Recurse -ErrorAction Stop)
    $reparse = @($children | Where-Object {
        $_.Attributes -band [IO.FileAttributes]::ReparsePoint
    })
    if ($reparse.Count -ne 0) {
        throw "Refusing candidate containing a reparse point: $($reparse[0].FullName)"
    }

    $nestedGit = @($children | Where-Object { $_.Name -eq '.git' })
    if ($nestedGit.Count -ne 0) {
        throw "Refusing candidate containing nested Git metadata: $($nestedGit[0].FullName)"
    }

    $files = @($children | Where-Object { -not $_.PSIsContainer })
    $directories = @($children | Where-Object { $_.PSIsContainer })
    $logicalBytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
    $rows = foreach ($item in ($children | Sort-Object FullName)) {
        $relative = [IO.Path]::GetRelativePath($canonical, $item.FullName).Replace('\', '/')
        $length = if ($item.PSIsContainer) { 0L } else { [int64]$item.Length }
        "$relative`t$length`t$($item.LastWriteTimeUtc.Ticks)`t$([int]$item.Attributes)"
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    $fingerprint = [Convert]::ToHexString(
        $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes(($rows -join "`n")))
    ).ToLowerInvariant()

    return [ordered]@{
        path = $canonical
        file_count = $files.Count
        directory_count = $directories.Count
        logical_bytes = $logicalBytes
        inventory_sha256 = $fingerprint
        latest_write_utc = @($children + $rootItem |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1).LastWriteTimeUtc.ToString('o')
    }
}

if ($PSCmdlet.ParameterSetName -eq 'Inventory') {
    Get-Inventory -Path $InventoryPath | ConvertTo-Json -Depth 4
    return
}

$manifestPath = Get-CanonicalExistingPath -Path $Manifest
$configuration = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($configuration.schema_version -ne 1) {
    throw "Unsupported cleanup manifest schema: $($configuration.schema_version)"
}
if (@($configuration.allowed_roots).Count -eq 0) {
    throw 'Cleanup manifest has no allowlisted roots.'
}
if (@($configuration.candidates).Count -eq 0) {
    throw 'Cleanup manifest has no candidates.'
}

$allowedRoots = @($configuration.allowed_roots | ForEach-Object {
    Get-CanonicalExistingPath -Path ([string]$_)
})
$protectedPaths = @($configuration.protected_paths | ForEach-Object {
    [IO.Path]::GetFullPath([string]$_).TrimEnd('\', '/')
})

if (-not $LogPath) {
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $LogPath = Join-Path (Split-Path -Parent $manifestPath) "cleanup-$stamp.jsonl"
}
$canonicalLog = [IO.Path]::GetFullPath($LogPath)
$logParent = Split-Path -Parent $canonicalLog
if (-not (Test-Path -LiteralPath $logParent -PathType Container)) {
    New-Item -ItemType Directory -Path $logParent | Out-Null
}
if (Test-Path -LiteralPath $canonicalLog) {
    throw "Refusing to overwrite cleanup log: $canonicalLog"
}

function Write-ActionLog {
    param([Parameter(Mandatory)] [hashtable] $Record)

    $line = $Record | ConvertTo-Json -Compress -Depth 8
    [IO.File]::AppendAllText($canonicalLog, $line + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
}

$mode = if ($Apply) { 'apply' } else { 'dry-run' }
$deletedBytes = 0L
$deletedCount = 0

foreach ($candidate in @($configuration.candidates)) {
    $started = [DateTime]::UtcNow
    $requestedPath = [string]$candidate.path
    try {
        $canonical = Get-CanonicalExistingPath -Path $requestedPath
        $matchedRoot = @($allowedRoots | Where-Object {
            Test-PathWithin -Path $canonical -Root $_
        })
        if ($matchedRoot.Count -eq 0) {
            throw "Candidate is outside every allowlisted root: $canonical"
        }
        foreach ($protected in $protectedPaths) {
            if ((Test-PathWithin -Path $canonical -Root $protected -AllowEqual) -or
                (Test-PathWithin -Path $protected -Root $canonical -AllowEqual)) {
                throw "Candidate overlaps protected path '$protected': $canonical"
            }
        }

        $before = Get-Inventory -Path $canonical
        foreach ($field in @('file_count', 'directory_count', 'logical_bytes', 'inventory_sha256')) {
            if ([string]$before[$field] -cne [string]$candidate.$field) {
                throw "Candidate changed since manifest creation ($field): $canonical"
            }
        }
        if ($candidate.PSObject.Properties.Name -contains 'content_sha256' -and
            -not [string]::IsNullOrWhiteSpace([string]$candidate.content_sha256)) {
            $item = Get-Item -LiteralPath $canonical -Force
            if ($item.PSIsContainer) {
                throw "content_sha256 is supported only for file candidates: $canonical"
            }
            $contentHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $canonical).Hash.ToLowerInvariant()
            if ($contentHash -cne ([string]$candidate.content_sha256).ToLowerInvariant()) {
                throw "Candidate content hash changed since manifest creation: $canonical"
            }
        }

        if ($Apply) {
            $immediate = Get-Inventory -Path $canonical
            if (($immediate | ConvertTo-Json -Compress) -cne ($before | ConvertTo-Json -Compress)) {
                throw "Candidate changed during pre-delete revalidation: $canonical"
            }
            Remove-Item -LiteralPath $canonical -Recurse -Force -ErrorAction Stop
            if (Test-Path -LiteralPath $canonical) {
                throw "Deletion returned without removing candidate: $canonical"
            }
            $deletedBytes += [int64]$before.logical_bytes
            $deletedCount++
        }

        Write-ActionLog -Record ([ordered]@{
            utc = [DateTime]::UtcNow.ToString('o')
            mode = $mode
            status = if ($Apply) { 'deleted' } else { 'validated' }
            path = $canonical
            owner_project = [string]$candidate.owner_project
            reason = [string]$candidate.reason
            preservation_action = [string]$candidate.preservation_action
            decision = [string]$candidate.decision
            logical_bytes = [int64]$before.logical_bytes
            file_count = [int]$before.file_count
            inventory_sha256 = [string]$before.inventory_sha256
            content_sha256 = if ($candidate.PSObject.Properties.Name -contains 'content_sha256') {
                [string]$candidate.content_sha256
            } else { $null }
            elapsed_seconds = ([DateTime]::UtcNow - $started).TotalSeconds
        })
    } catch {
        Write-ActionLog -Record ([ordered]@{
            utc = [DateTime]::UtcNow.ToString('o')
            mode = $mode
            status = 'failed'
            path = $requestedPath
            error = $_.Exception.Message
            elapsed_seconds = ([DateTime]::UtcNow - $started).TotalSeconds
        })
        throw
    }
}

Write-ActionLog -Record ([ordered]@{
    utc = [DateTime]::UtcNow.ToString('o')
    mode = $mode
    status = 'complete'
    candidate_count = @($configuration.candidates).Count
    deleted_count = $deletedCount
    logical_bytes_deleted = $deletedBytes
})

Write-Output "mode=$mode"
Write-Output "candidates=$(@($configuration.candidates).Count)"
Write-Output "deleted=$deletedCount"
Write-Output "logical_bytes_deleted=$deletedBytes"
Write-Output "log=$canonicalLog"
