$ErrorActionPreference = 'Stop'
# Bootstrap is CPU-only. Constrain this shell before compiling the Win32 binding.
$cleanupProcess = Get-Process -Id $PID
$cleanupProcess.ProcessorAffinity = 255
$cleanupProcess.PriorityClass = 'BelowNormal'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HIP_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:CMAKE_BUILD_PARALLEL_LEVEL = '4'
$env:CL = '/MP1'
$env:NVCC_PREPEND_FLAGS = '--threads 1'
$env:VOIDINFER_PARITY_DFLASH_TIMING = '0'
if (-not ('CleanupJob' -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class CleanupJob {
    [StructLayout(LayoutKind.Sequential)] struct Basic {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinWS, MaxWS;
        public uint ActiveProcesses;
        public UIntPtr Affinity;
        public uint Priority, Scheduling;
    }
    [StructLayout(LayoutKind.Sequential)] struct IO {
        public ulong ReadOps, WriteOps, OtherOps, ReadBytes, WriteBytes, OtherBytes;
    }
    [StructLayout(LayoutKind.Sequential)] struct Extended {
        public Basic Basic;
        public IO IO;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern IntPtr CreateJobObject(IntPtr security, string name);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool SetInformationJobObject(IntPtr job, int info, ref Extended limits, uint size);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")] static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);
    public static IntPtr Handle;
    public static string Name;
    public static void Enter() {
        if (Handle != IntPtr.Zero) return;
        Name = "Local\\VoidInferCleanup-" + Guid.NewGuid().ToString("N");
        Handle = CreateJobObject(IntPtr.Zero, Name);
        if (Handle == IntPtr.Zero) throw new Win32Exception();
        var limits = new Extended();
        // Affinity, priority, active-process limit, kill-on-close; NO breakaway flags.
        limits.Basic.Flags = 0x10 | 0x20 | 0x8 | 0x2000;
        limits.Basic.Affinity = new UIntPtr(255);
        limits.Basic.Priority = 0x4000; // BELOW_NORMAL_PRIORITY_CLASS
        limits.Basic.ActiveProcesses = 8;
        if (!SetInformationJobObject(Handle, 9, ref limits, (uint)Marshal.SizeOf(limits)))
            throw new Win32Exception();
        if (!AssignProcessToJobObject(Handle, GetCurrentProcess())) throw new Win32Exception();
    }
}
'@
}
[CleanupJob]::Enter()
$env:VOIDINFER_CLEANUP_JOB_NAME = [CleanupJob]::Name
