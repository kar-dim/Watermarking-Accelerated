@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "CLANG_FORMAT=%SCRIPT_DIR%clang-format.exe"

if not exist "%CLANG_FORMAT%" (
    echo [Watermarking] Error: clang-format.exe not found in %SCRIPT_DIR%
    exit /b 1
)

if "%~1"=="" (
    set "PWSH_PATHS='Watermarking-CLI', 'Watermarking-Core', 'Watermarking-Core-tests', 'Watermarking-UI'"
) else (
    set "PWSH_PATHS='%~1'"
)

rem one clang-format process per batch of files (not one per file), batches keep the command line under the Windows limit
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; Set-Location -LiteralPath '%SCRIPT_DIR%'; $files = @(Get-ChildItem -Path %PWSH_PATHS% -Recurse -File -Include *.h, *.hpp, *.cpp, *.c, *.cu, *.cuh, *.cl | Where-Object { $_.FullName -notmatch '\\(libs|vendor|third_party|x64)\\' -and $_.Name -notmatch '^TinyEXIF\.(cpp|h)$' } | ForEach-Object { $_.FullName }); for ($first = 0; $first -lt $files.Count; $first += 100) { $batch = $files[$first..([Math]::Min($first + 99, $files.Count - 1))]; & '%CLANG_FORMAT%' -i --style=file $batch; if ($LASTEXITCODE -ne 0) { throw \"clang-format failed for one of: $($batch -join ', ')\" } }"

if errorlevel 1 exit /b %errorlevel%
endlocal
