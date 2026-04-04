@echo off
set MSVC_BASE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
set SDK_BASE=C:\Program Files (x86)\Windows Kits\10
set SDK_VER=10.0.26100.0

set PATH=%MSVC_BASE%\bin\Hostx64\x64;%SDK_BASE%\bin\%SDK_VER%\x64;%PATH%
set LIB=%MSVC_BASE%\lib\x64;%SDK_BASE%\Lib\%SDK_VER%\um\x64;%SDK_BASE%\Lib\%SDK_VER%\ucrt\x64
set INCLUDE=%MSVC_BASE%\include;%SDK_BASE%\Include\%SDK_VER%\ucrt;%SDK_BASE%\Include\%SDK_VER%\um;%SDK_BASE%\Include\%SDK_VER%\shared

set CMAKE_GENERATOR=Ninja
set CMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe
set CMAKE_PREFIX_PATH=D:\Projects\2026\hackathon\test\Depth-Anything-V2\.venv\Lib\site-packages\torch

echo LIB=%LIB%
echo Running pip install...
"D:\Projects\2026\hackathon\test\Depth-Anything-V2\.venv\Scripts\pip.exe" install git+https://github.com/tatsy/torchmcubes.git
