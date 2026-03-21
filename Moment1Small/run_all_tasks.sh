#!/usr/bin/env bash
set -e

PYTHON="/workspace/ruiyang/temp_parkinsons/.venv/bin/python"
SCRIPT="/workspace/ruiyang/temp_parkinsons/parkinsons-disease-smartwatch/moment_training/train_moment_pd_vs_hc.py"
RUN_ROOT="/workspace/ruiyang/temp_parkinsons/parkinsons-disease-smartwatch/moment_training/runs_pd_hc_batch"
OUT_ROOT="/workspace/ruiyang/temp_parkinsons/parkinsons-disease-smartwatch/moment_training/outputs_pd_hc_batch"
CSV_ROOT="/workspace/ruiyang/temp_parkinsons/parkinsons-disease-smartwatch/data/out"

mkdir -p "${RUN_ROOT}" "${OUT_ROOT}" "${CSV_ROOT}"

run_job() {
  local name="$1"
  shift
  echo "==> Running ${name}"
  ${PYTHON} ${SCRIPT} \
    --log-dir "${RUN_ROOT}/${name}" \
    --output-dir "${OUT_ROOT}/${name}" \
    --out-csv "${CSV_ROOT}/${name}.csv" \
    "$@"
}

# 1-6: PD vs HC
run_job "pd_hc_preprocessed_both" --task binary --source preprocessed --sensor-filter both
run_job "pd_hc_preprocessed_acceleration" --task binary --source preprocessed --sensor-filter acceleration
run_job "pd_hc_preprocessed_rotation" --task binary --source preprocessed --sensor-filter rotation
run_job "pd_hc_raw_both" --task binary --source raw --sensor-filter both
run_job "pd_hc_raw_acceleration" --task binary --source raw --sensor-filter acceleration
run_job "pd_hc_raw_rotation" --task binary --source raw --sensor-filter rotation

# PD vs other (dd)
run_job "pd_dd_preprocessed_both" --task pd_dd --source preprocessed --sensor-filter both
run_job "pd_dd_preprocessed_acceleration" --task pd_dd --source preprocessed --sensor-filter acceleration
run_job "pd_dd_preprocessed_rotation" --task pd_dd --source preprocessed --sensor-filter rotation
run_job "pd_dd_raw_both" --task pd_dd --source raw --sensor-filter both
run_job "pd_dd_raw_acceleration" --task pd_dd --source raw --sensor-filter acceleration
run_job "pd_dd_raw_rotation" --task pd_dd --source raw --sensor-filter rotation

# 7-12: HC/PD/other (dd)
run_job "hc_pd_other_preprocessed_both" --task multiclass --source preprocessed --sensor-filter both
run_job "hc_pd_other_preprocessed_acceleration" --task multiclass --source preprocessed --sensor-filter acceleration
run_job "hc_pd_other_preprocessed_rotation" --task multiclass --source preprocessed --sensor-filter rotation
run_job "hc_pd_other_raw_both" --task multiclass --source raw --sensor-filter both
run_job "hc_pd_other_raw_acceleration" --task multiclass --source raw --sensor-filter acceleration
run_job "hc_pd_other_raw_rotation" --task multiclass --source raw --sensor-filter rotation

echo "All jobs finished."
