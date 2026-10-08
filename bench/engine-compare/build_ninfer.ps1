# Native MSVC + CUDA 13.3 build of the NInfer Windows port (upstream 81c8ce0 merged).
$ErrorActionPreference='Stop'
$src='D:\AI\ninfer-5090-windows'
$build="$src\build"
$lines=& $env:ComSpec /d /s /c "`"C:\BuildTools2026\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 >nul 2>nul && set"
foreach($line in $lines){ if($line -match '^([^=]+)=(.*)$'){ [Environment]::SetEnvironmentVariable($Matches[1],$Matches[2],'Process') } }
$cuda='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$env:CUDA_PATH=$cuda; $env:PATH="$cuda\bin;"+$env:PATH
$cmake='C:\BuildTools2026\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ninja='C:\BuildTools2026\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
& $cmake -S $src -B $build -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_CUDA_COMPILER=$cuda\bin\nvcc.exe" `
    "-DCUDAToolkit_ROOT=$cuda" '-DCMAKE_CUDA_ARCHITECTURES=120a' '-DNINFER_ENABLE_AVX2=ON' '-DCMAKE_BUILD_TYPE=Release' *> "$src\configure.log"
if($LASTEXITCODE -ne 0){ Get-Content "$src\configure.log" -Tail 30; throw 'configure failed' }
& $cmake --build $build -j --target ninfer ninfer-serve ninfer-perplexity *> "$src\build.log"
if($LASTEXITCODE -ne 0){ Select-String "$src\build.log" -Pattern 'error|FAILED' | Select -First 15 | % Line; throw 'build failed' }
foreach($d in 'avcodec','avformat','avutil','swscale','swresample'){ Copy-Item "$src\ffmpeg\bin\$d-*.dll" "$build\apps\" -Force }
Get-ChildItem "$build\apps\*.exe" | Select Name,Length,LastWriteTime
