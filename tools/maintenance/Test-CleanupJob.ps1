param([ValidateRange(0, 2)] [int] $Depth = 0)
$ErrorActionPreference = 'Stop'
if ($Depth -eq 0) { . (Join-Path $PSScriptRoot 'Enter-CleanupJob.ps1') }
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class CleanupJobProbe {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    public static extern IntPtr OpenJobObject(uint access, bool inherit, string name);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);
    [DllImport("kernel32.dll")] public static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetProcessAffinityMask(IntPtr process, UIntPtr mask);
    [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr handle);
}
'@
$job = [CleanupJobProbe]::OpenJobObject(4, $false, $env:VOIDINFER_CLEANUP_JOB_NAME)
if ($job -eq [IntPtr]::Zero) { throw 'Cannot open the campaign Job Object' }
try {
    $inside = $false
    if (-not [CleanupJobProbe]::IsProcessInJob([CleanupJobProbe]::GetCurrentProcess(), $job, [ref] $inside) -or -not $inside) {
        throw 'Process escaped the campaign Job Object'
    }
    $process = Get-Process -Id $PID
    if ($process.ProcessorAffinity.ToInt64() -ne 255 -or $process.PriorityClass -ne 'BelowNormal') {
        throw 'Affinity/priority constraint mismatch'
    }
    # Windows may report success while keeping the job's effective mask. Check
    # the actual process mask, not the setter's return value.
    [void] [CleanupJobProbe]::SetProcessAffinityMask([CleanupJobProbe]::GetCurrentProcess(), [UIntPtr]::new(511))
    $process.Refresh()
    if ($process.ProcessorAffinity.ToInt64() -ne 255) {
        # Restore immediately before reporting a containment failure.
        $process.ProcessorAffinity = 255
        throw 'Job allowed widening beyond eight logical processors'
    }
    Write-Output "PASS depth=$Depth PID=$PID job-member affinity=0xFF priority=BelowNormal widening-ineffective"
    if ($Depth -lt 2) {
        & (Join-Path $PSHOME 'pwsh.exe') -NoProfile -File $PSCommandPath -Depth ($Depth + 1)
        if ($LASTEXITCODE -ne 0) { throw 'Descendant containment check failed' }
    }
} finally {
    [void] [CleanupJobProbe]::CloseHandle($job)
}
