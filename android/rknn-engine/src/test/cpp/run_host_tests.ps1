$ErrorActionPreference = "Stop"

$testRoot = $PSScriptRoot
$engineRoot = Resolve-Path (Join-Path $testRoot "..\..\..")
$compiler = if ($env:CXX) { $env:CXX } else { "g++" }
$buildDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "rknn-engine-host-tests"
$executable = Join-Path $buildDirectory "rknn_yolov8_output_processor_test.exe"

New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null

& $compiler `
    -std=c++17 `
    -Wall `
    -Wextra `
    -Wpedantic `
    -Werror `
    -I (Join-Path $testRoot "stubs") `
    -I (Join-Path $engineRoot "src\main\cpp") `
    -I (Join-Path $engineRoot "src\main\cpp\third_party\rknn") `
    (Join-Path $testRoot "rknn_yolov8_output_processor_test.cpp") `
    (Join-Path $engineRoot "src\main\cpp\rknn\rknn_yolov8_output_processor.cpp") `
    -o $executable

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $executable
exit $LASTEXITCODE
