param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][ValidateSet('validation','test')][string]$Split,
    [Parameter(Mandatory=$true)][string]$OutputPrefix,
    [string]$Python = 'python'
)

$outputPath = "$OutputPrefix.json"
$stdoutPath = "$OutputPrefix.stdout.log"
$stderrPath = "$OutputPrefix.stderr.log"
if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath
}
$process = Start-Process -FilePath $Python -ArgumentList @(
    'evaluate.py', '--checkpoint', $Checkpoint, '--device', 'cpu',
    '--precision', 'fp32', '--threads', '4', '--split', $Split,
    '--output', $outputPath
) -WorkingDirectory (Get-Location).Path -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -WindowStyle Hidden -PassThru
$peakBytes = 0L
while (-not $process.HasExited) {
    $process.Refresh()
    $peakBytes = [Math]::Max($peakBytes, $process.PeakWorkingSet64)
    Start-Sleep -Milliseconds 100
}
$process.WaitForExit()
$process.Refresh()
$peakBytes = [Math]::Max($peakBytes, $process.PeakWorkingSet64)
if (($null -ne $process.ExitCode -and $process.ExitCode -ne 0) -or
    -not (Test-Path -LiteralPath $outputPath)) {
    throw "Evaluation failed with exit code $($process.ExitCode). See $stderrPath"
}
$evaluation = Get-Content -LiteralPath $outputPath -Raw | ConvertFrom-Json
$resource = [ordered]@{
    checkpoint = $Checkpoint
    split = $Split
    bpb = $evaluation.bpb
    scorer_seconds = $evaluation.seconds
    sampled_peak_working_set_gib = $peakBytes / 1GB
    checkpoint_sha256 = $evaluation.checkpoint_sha256
}
$resourcePath = "$OutputPrefix.resource.json"
$resource | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $resourcePath
$resource | ConvertTo-Json -Depth 4
