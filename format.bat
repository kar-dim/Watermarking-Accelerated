@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "CLANG_FORMAT=%SCRIPT_DIR%clang-format.exe"

if not exist "%CLANG_FORMAT%" (
    echo [Watermarking] Error: clang-format.exe not found in %SCRIPT_DIR%
    exit /b 1
)

if "%~1"=="" (
    set "PWSH_PATHS='Watermarking-CLI', 'Watermarking-Impl', 'Watermarking-Impl-tests', 'Watermarking-BenchUI', 'Watermarking-Util'"
) else (
    set "PWSH_PATHS='%~1'"
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; Set-Location -LiteralPath '%SCRIPT_DIR%'; $files = Get-ChildItem -Path %PWSH_PATHS% -Recurse -File -Include *.h, *.hpp, *.cpp, *.c, *.cu, *.cuh, *.cl | Where-Object { $_.FullName -notmatch '\\(libs|vendor|third_party|x64)\\' -and $_.Name -notmatch '^TinyEXIF\.(cpp|h)$' }; foreach ($f in $files) { & '%CLANG_FORMAT%' -i --style=file $f.FullName; if ($LASTEXITCODE -ne 0) { throw \"clang-format failed for $($f.FullName)\" } }"

if errorlevel 1 exit /b %errorlevel%
endlocal
