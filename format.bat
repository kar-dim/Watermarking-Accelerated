@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "CLANG_FORMAT=%SCRIPT_DIR%clang-format.exe"

if not exist "%CLANG_FORMAT%" (
    echo [Watermarking] Error: clang-format.exe not found in %SCRIPT_DIR%
    exit /b 1
)

set "FORMAT_TARGET=%~1"

rem one clang-format process per batch of files (not one per file), batches keep the command line under the Windows limit
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; Set-Location -LiteralPath $env:SCRIPT_DIR; $roots = if ($env:FORMAT_TARGET) { $env:FORMAT_TARGET } else { 'Watermarking-CLI', 'Watermarking-Core', 'Watermarking-Core-tests', 'Watermarking-UI' }; $files = @(Get-ChildItem -LiteralPath $roots -Recurse -File | Where-Object { $_.Extension -in '.h','.hpp','.cpp','.c','.cu','.cuh','.cl' -and $_.FullName.Substring($env:SCRIPT_DIR.Length - 1) -notmatch '\\(libs|vendor|third_party|x64)\\' -and $_.Name -notmatch '^TinyEXIF\.(cpp|h)$' } | ForEach-Object { $_.FullName }); for ($first = 0; $first -lt $files.Count; $first += 100) { $batch = $files[$first..([Math]::Min($first + 99, $files.Count - 1))]; & $env:CLANG_FORMAT -i --style=file $batch; if ($LASTEXITCODE -ne 0) { throw \"clang-format failed for one of: $($batch -join ', ')\" } }"

if errorlevel 1 exit /b %errorlevel%
endlocal
