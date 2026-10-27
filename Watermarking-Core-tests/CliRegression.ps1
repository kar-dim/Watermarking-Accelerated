# Exercise argument parsing and batch exit codes against a built CLI executable.
param([string]$Executable = 'x64/EIGEN_Release/Watermarking-CLI.exe')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$executablePath = if ([IO.Path]::IsPathRooted($Executable)) { $Executable } else { Join-Path $taskRoot $Executable }
$cliPath = (Resolve-Path -LiteralPath $executablePath).Path
# Keep generated outputs and logs together for inspection.
$fixtureDirectory = Join-Path $taskRoot ('x64/review/cli-regression-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $fixtureDirectory | Out-Null
$imagePath = Join-Path $taskRoot 'samples/images/512.png'
$baseArguments = @('--watermark_password', 'regression-password', '--image.path', $imagePath, '--output_path', (Join-Path $fixtureDirectory 'output.png'))
$passed = 0
# Close stdin and enforce a timeout so an interactive pause fails the case.
function Invoke-Case([string]$Name, [string[]]$Arguments, [int]$ExpectedExit, [string]$ExpectedText) {
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $cliPath
    $start.WorkingDirectory = $fixtureDirectory
    $start.UseShellExecute = $false
    $start.RedirectStandardInput = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in $Arguments) { $start.ArgumentList.Add($argument) }
    $process = [Diagnostics.Process]::Start($start)
    $process.StandardInput.Close()
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    if (!$process.WaitForExit(30000)) { $process.Kill(); throw "$Name did not exit without user input" }
    $text = $stdout.Result + $stderr.Result
    Set-Content -LiteralPath (Join-Path $fixtureDirectory "$Name.log") -Value $text
    if ($process.ExitCode -ne $ExpectedExit -or $text -notmatch $ExpectedText -or $text -match 'Press any key') {
        throw "$Name failed: exit $($process.ExitCode), expected $ExpectedExit; output: $text"
    }
    $script:passed++
    Write-Output "PASS $Name"
    $process.Dispose()
}

# An isolated working directory verifies that commands need no external configuration.
Invoke-Case 'no-arguments-help' @() 0 'Usage: Watermarking-CLI'
Invoke-Case 'help' @('--help') 0 'Usage: Watermarking-CLI'
Invoke-Case 'valid-defaults' $baseArguments 0 'Embedded watermark.*p = 3, PSNR = 40'
Invoke-Case 'explicit-options' ($baseArguments + @('--p', '5', '--psnr', '42', '--no-pause')) 0 'Embedded watermark.*p = 5, PSNR = 42'
Invoke-Case 'qualified-and-equals' @('--global.watermark_password=regression-password', '--image.mode=single', "--image.path=$imagePath", "--image.output_path=$fixtureDirectory/output.png", '--global.p=5', '--global.psnr=42') 0 'Embedded watermark.*p = 5, PSNR = 42'
Invoke-Case 'repeated-option' ($baseArguments + @('--p', 'invalid', '--p', '3')) 0 'Embedded watermark'
Invoke-Case 'explicit-gpu-device' ($baseArguments + @('--gpu_device_id', '0')) 0 'Embedded watermark'
# Required inputs should produce actionable errors instead of implicit sample operations.
Invoke-Case 'missing-password' @('--image.path', $imagePath, '--output_path', "$fixtureDirectory/output.png") 1 'No valid watermark password'
Invoke-Case 'missing-output' @('--watermark_password', 'regression-password', '--image.path', $imagePath) 1 'requires --output_path'
Invoke-Case 'missing-input' @('--watermark_password', 'regression-password', '--output_path', "$fixtureDirectory/output.png") 1 'No valid image file'
Invoke-Case 'unknown-option' @('--not-an-option', 'value') 1 'Unknown command-line option'
Invoke-Case 'missing-value' @('--psnr') 1 'Missing value'
Invoke-Case 'ambiguous-option' @('--mode', 'single') 1 'Ambiguous setting'
Invoke-Case 'positional-argument' @('unexpected') 1 'Unexpected positional argument'
# Numeric options must reject trailing text and out-of-range values.
foreach ($value in @('3garbage', '999999999999999999999999')) {
    $name = if ($value -eq '3garbage') { 'trailing' } else { 'overflow' }
    Invoke-Case "cli-integer-$name" ($baseArguments + @('--p', $value)) 1 'Invalid integer'
}
foreach ($value in @('40garbage', 'nan', 'inf', '1e50')) {
    $name = if ($value -eq '40garbage') { 'trailing' } else { $value }
    Invoke-Case "cli-number-$name" ($baseArguments + @('--psnr', $value)) 1 'Invalid number'
}
# Invalid booleans should fail parsing before a missing video is opened.
Invoke-Case 'cli-boolean' @('--watermark_password', 'regression-password', '--video.path', 'missing-video.mkv', '--hw_encoder', 'maybe') 1 'Invalid boolean'
Invoke-Case 'removed-cuda-encoder-option' @('--cuda_hw_encoder', 'true') 1 'Unknown command-line option'
# Benchmark defaults must reach sample loading without requiring a password option.
Invoke-Case 'benchmark-defaults' @('--bench', '--benchmark_loops', '1') 1 'Benchmark image not found'
Invoke-Case 'benchmark-invalid-loops' @('--bench', '--benchmark_loops', '0') 1 'benchmark_loops must be positive'
# A corrupt image must make a partially successful batch return a failure code.
$batchDirectory = Join-Path $fixtureDirectory 'batch'
New-Item -ItemType Directory -Path $batchDirectory | Out-Null
Copy-Item -LiteralPath $imagePath -Destination (Join-Path $batchDirectory 'valid.png')
Set-Content -LiteralPath (Join-Path $batchDirectory 'invalid.png') -Value 'corrupt image'
foreach ($mode in @('batch_detect', 'batch_embed')) {
    Invoke-Case $mode ($baseArguments + @('--image.mode', $mode, '--image.path', $batchDirectory)) 1 'Successfully processed 1/2'
}
# The same batch should succeed once every input is valid.
Remove-Item -LiteralPath (Join-Path $batchDirectory 'invalid.png')
Invoke-Case 'complete-batch' ($baseArguments + @('--image.mode', 'batch_detect', '--image.path', $batchDirectory)) 0 'Successfully processed 1/1'
Invoke-Case 'image-batch-device' ($baseArguments + @('--image.mode', 'batch_detect', '--image.path', $batchDirectory, '--image.devices', '0')) 0 'Successfully processed 1/1'
Invoke-Case 'image-batch-duplicate-devices' ($baseArguments + @('--image.mode', 'batch_detect', '--image.path', $batchDirectory, '--image.devices', '0,0')) 1 'selected once'
Invoke-Case 'image-batch-unavailable-device' ($baseArguments + @('--image.mode', 'batch_detect', '--image.path', $batchDirectory, '--image.devices', '99999')) 1 'selected image batch device is unavailable'
Invoke-Case 'image-batch-malformed-devices' ($baseArguments + @('--image.mode', 'batch_detect', '--image.path', $batchDirectory, '--image.devices', '0,')) 1 'device index is missing'
Invoke-Case 'image-devices-single-mode' ($baseArguments + @('--image.devices', '0')) 1 'applies to image batch modes'
Invoke-Case 'video-devices-single-mode' ($baseArguments + @('--video.devices', '0')) 1 'applies to video batch modes'
Invoke-Case 'ambiguous-devices-option' @('--devices', '0') 1 'Ambiguous setting'
Invoke-Case 'no-interactive-pause' ($baseArguments + @('--image.mode', 'invalid')) 1 'Invalid mode'
Write-Output "$passed CLI regression cases passed. Logs: $fixtureDirectory"
