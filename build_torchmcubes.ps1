$MSVC_BASE = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
$SDK_BASE = "C:\Program Files (x86)\Windows Kits\10"
$SDK_VER = "10.0.26100.0"

$env:PATH = "$MSVC_BASE\bin\Hostx64\x64;$SDK_BASE\bin\$SDK_VER\x64;" + $env:PATH
$env:LIB = "$MSVC_BASE\lib\x64;$SDK_BASE\Lib\$SDK_VER\um\x64;$SDK_BASE\Lib\$SDK_VER\ucrt\x64"
$env:INCLUDE = "$MSVC_BASE\include;$SDK_BASE\Include\$SDK_VER\ucrt;$SDK_BASE\Include\$SDK_VER\um;$SDK_BASE\Include\$SDK_VER\shared"
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_CUDA_COMPILER = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
$env:CMAKE_PREFIX_PATH = "D:\Projects\2026\hackathon\test\Depth-Anything-V2\.venv\Lib\site-packages\torch"
$env:CUDAFLAGS = "-allow-unsupported-compiler"
$env:CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
$env:SKBUILD_CMAKE_ARGS = "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"

Write-Host "LIB=$env:LIB"
Write-Host "PATH includes: $MSVC_BASE\bin\Hostx64\x64"

& "D:\Projects\2026\hackathon\test\Depth-Anything-V2\.venv\Scripts\pip.exe" install "D:\Projects\2026\hackathon\test\torchmcubes_src"
