param(
    [string]$Python = "python"
)

$RepoRoot = Resolve-Path "${PSScriptRoot}\.."
$Script = Join-Path $RepoRoot "scripts\grid_search_cnn_3class.py"

$common = @("--force-rebuild")
$sensors = @("both", "acceleration", "rotation")

foreach ($sensor in $sensors) {
    $sensorArgs = @("--sensor-filter", $sensor)

    # 1) raw data - 3 class
    & $Python $Script --task 3class --source raw --raw-mode split @sensorArgs @common
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # 2) preprocessed data - 3 class
    & $Python $Script --task 3class --source preprocessed @sensorArgs @common
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # 3) raw data - HC vs PD
    & $Python $Script --task binary --binary-mode pd_vs_hc --source raw --raw-mode split @sensorArgs @common
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # 4) preprocessed data - HC vs PD
    & $Python $Script --task binary --binary-mode pd_vs_hc --source preprocessed @sensorArgs @common
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # 5) raw data - PD vs DD
    & $Python $Script --task binary --binary-mode pd_vs_dd --source raw --raw-mode split @sensorArgs @common
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # 6) preprocessed data - PD vs DD
    & $Python $Script --task binary --binary-mode pd_vs_dd --source preprocessed @sensorArgs @common
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
