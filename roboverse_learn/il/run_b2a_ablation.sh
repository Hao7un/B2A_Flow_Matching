#!/usr/bin/env bash
set -euo pipefail

# B2A benchmark grid.
#
# Models:
#   - kalman_b2a: Kalman-B2A posterior belief source
#
# Extra supported methods for manual overrides:
#   - a2a_noise_0p02: A2A-Noise with history_noise_std=0.02
#   - b2a_v0:         B2A amortized belief source
#   - b2a_v1:         B2A recursive belief source
#   - kalman_b2a_v2:  Kalman-B2A + decoupled source noise (0.1, train & eval),
#                     stochastic eval, no W2 sigma penalty
#
# Default benchmark grid:
#   models       : kalman_b2a
#   tasks        : close_box pick_cube
#   train/eval NFE: 1 6 9
#   flow matcher : ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
#   eval DR      : 0 1 2 3
#
# Output layout is intentionally human-readable:
#   ./il_outputs/b2a_bench/<RUN_TAG>/<matcher>/<method>/nfe<N>/<task>/...
#   ./il_outputs/b2a_bench/<RUN_TAG>/<matcher>/<method>/nfe<N>/<task>/eval/...
#   ./il_outputs/summaries/<RUN_TAG>/results.csv
#
# Usage from repo root:
#   bash roboverse_learn/il/run_b2a_ablation.sh
#
# Useful overrides:
#   GPU=1 SIM=isaacsim RUN_TAG=my_benchmark bash roboverse_learn/il/run_b2a_ablation.sh
#   RUN_TRAIN=0 RUN_EVAL=1 RUN_TAG=my_benchmark bash roboverse_learn/il/run_b2a_ablation.sh
#   TASK_LIST="close_box" NFE_LIST="1 6" MATCHER_LIST="cfm" bash roboverse_learn/il/run_b2a_ablation.sh
#
# Backward compatibility:
#   RUN_PREFIX is accepted as an alias for RUN_TAG.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

clean_pythonpath() {
  local existing="${PYTHONPATH:-}"
  local cleaned=""
  local entry
  IFS=":" read -r -a entries <<< "${existing}"
  for entry in "${entries[@]}"; do
    [[ -z "${entry}" ]] && continue
    [[ "${entry}" == "${REPO_ROOT}" ]] && continue
    [[ "${entry}" == *"/MoS-Flow"* ]] && continue
    if [[ -z "${cleaned}" ]]; then
      cleaned="${entry}"
    else
      cleaned="${cleaned}:${entry}"
    fi
  done
  echo "${cleaned}"
}

CLEANED_PYTHONPATH="$(clean_pythonpath)"
if [[ -n "${CLEANED_PYTHONPATH}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${CLEANED_PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi
export METASIM_FORCE_EXIT_ON_CLOSE=1
export METASIM_CLOSE_TIMEOUT_SEC="${METASIM_CLOSE_TIMEOUT_SEC:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"

TASK_LIST="${TASK_LIST:-close_box pick_cube}"
METHOD_LIST="${METHOD_LIST:-kalman_b2a}"
NFE_LIST="${NFE_LIST:-1 6}"
MATCHER_LIST="${MATCHER_LIST:-cfm exact_ot}"
DR_LIST="${DR_LIST:-0 1 2 3}"

read -r -a TASKS <<< "${TASK_LIST}"
read -r -a METHODS <<< "${METHOD_LIST}"
read -r -a NFE_STEPS <<< "${NFE_LIST}"
read -r -a MATCHERS <<< "${MATCHER_LIST}"
read -r -a DR_LEVELS <<< "${DR_LIST}"

GPU="${GPU:-0}"
SIM="${SIM:-isaacsim}"
DEMO_NUM="${DEMO_NUM:-100}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"
EVAL_SEED="${EVAL_SEED:-42}"
OFFSET_EVAL_SEED_BY_DR="${OFFSET_EVAL_SEED_BY_DR:-0}"
OBS_SPACE="${OBS_SPACE:-joint_pos}"
ACT_SPACE="${ACT_SPACE:-joint_pos}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:${GPU}}"
TRAIN_DR_LEVEL="${TRAIN_DR_LEVEL:-0}"
DR_SCENE_MODE="${DR_SCENE_MODE:-0}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
EVAL_MAX_STEP="${EVAL_MAX_STEP:-300}"
MAX_DEMO="${MAX_DEMO:-50}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RESUME_TRAIN="${RESUME_TRAIN:-1}"
SKIP_EXISTING_TRAIN="${SKIP_EXISTING_TRAIN:-1}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-1}"

TASK_LABEL="$(IFS=-; echo "${TASKS[*]}")"
MATCHER_LABEL="$(IFS=-; echo "${MATCHERS[*]}")"
NFE_LABEL="$(IFS=-; echo "${NFE_STEPS[*]}")"
DEFAULT_RUN_TAG="benchmark_${TASK_LABEL}_${DEMO_NUM}demo_${EPOCHS}epoch_seed${SEED}_nfe${NFE_LABEL}_${MATCHER_LABEL}_$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-${RUN_PREFIX:-${DEFAULT_RUN_TAG}}}"
EXP_GROUP="${EXP_GROUP:-bench/${RUN_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./il_outputs/b2a_bench/${RUN_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-./il_outputs/summaries/${RUN_TAG}}"
RESULTS_CSV="${RESULTS_CSV:-${RESULTS_DIR}/results.csv}"
MANIFEST="${RESULTS_DIR}/manifest.txt"
MAIN_SCRIPT="./roboverse_learn/il/train.py"
mkdir -p "${RESULTS_DIR}"

cat > "${MANIFEST}" <<EOF
run_tag: ${RUN_TAG}
exp_group: ${EXP_GROUP}
output_root: ${OUTPUT_ROOT}
tasks: ${TASKS[*]}
methods: ${METHODS[*]}
flow_matchers: ${MATCHERS[*]}
train_eval_nfe_steps: ${NFE_STEPS[*]}
train_demos: ${DEMO_NUM}
epochs: ${EPOCHS}
train_seed: ${SEED}
eval_seed_base: ${EVAL_SEED}
eval_seed_offset_by_dr: ${OFFSET_EVAL_SEED_BY_DR}
train_randomization_level: ${TRAIN_DR_LEVEL}
eval_randomization_levels: ${DR_LEVELS[*]}
sim: ${SIM}
obs_space: ${OBS_SPACE}
act_space: ${ACT_SPACE}
train_device: ${TRAIN_DEVICE}
eval_num_envs: ${EVAL_NUM_ENVS}
eval_max_step: ${EVAL_MAX_STEP}
max_demo: ${MAX_DEMO}
resume_train: ${RESUME_TRAIN}
skip_existing_train: ${SKIP_EXISTING_TRAIN}
skip_existing_eval: ${SKIP_EXISTING_EVAL}
train_output_pattern: ${OUTPUT_ROOT}/<matcher>/<method>/nfe<nfe>/<task>
eval_output_pattern: ${OUTPUT_ROOT}/<matcher>/<method>/nfe<nfe>/<task>/eval/<task>/<policy>/franka/randomization_level_<level>/scene_<scene>/seed_<seed>
EOF

echo "run_tag,method,policy,flow_matcher,flow_matcher_target,nfe,task,train_randomization_level,eval_randomization_level,scene,train_demos,epochs,seed,eval_seed,success_rate,stats_file" > "${RESULTS_CSV}"

debug_header() {
  echo ""
  echo "Run tag         : ${RUN_TAG}"
  echo "Output root     : ${OUTPUT_ROOT}"
  echo "Summary CSV     : ${RESULTS_CSV}"
  echo "Manifest        : ${MANIFEST}"
  echo "Experiment group: ${EXP_GROUP}"
  echo "Methods         : ${METHODS[*]}"
  echo "Tasks           : ${TASKS[*]}"
  echo "Flow matchers   : ${MATCHERS[*]}"
  echo "Train/eval NFE  : ${NFE_STEPS[*]}"
  echo "Resume train    : ${RESUME_TRAIN}"
  echo "Skip existing   : train=${SKIP_EXISTING_TRAIN}, eval=${SKIP_EXISTING_EVAL}"
}

method_policy() {
  case "$1" in
    a2a_noise_0p02) echo "a2a_noise" ;;
    b2a_v0) echo "b2a" ;;
    b2a_v1) echo "b2a_recursive" ;;
    kalman_b2a) echo "kalman_b2a" ;;
    kalman_b2a_v2) echo "kalman_b2a_v2" ;;
    *) echo "Unknown method: $1" >&2; return 1 ;;
  esac
}

method_overrides() {
  case "$1" in
    a2a_noise_0p02)
      echo "policy_config.history_noise_std=0.02"
      ;;
    b2a_v0|b2a_v1|kalman_b2a|kalman_b2a_v2)
      ;;
    *) echo "Unknown method: $1" >&2; return 1 ;;
  esac
}

matcher_target() {
  case "$1" in
    cfm|conditional)
      echo "roboverse_learn.il.utils.flow.flow_matchers.ConditionalFlowMatcher"
      ;;
    exact_ot|ot|exact|exact_optimal_transport)
      echo "roboverse_learn.il.utils.flow.flow_matchers.ExactOptimalTransportConditionalFlowMatcher"
      ;;
    *) echo "Unknown flow matcher: $1" >&2; return 1 ;;
  esac
}

matcher_name() {
  case "$1" in
    cfm|conditional) echo "conditional_fm" ;;
    exact_ot|ot|exact|exact_optimal_transport) echo "exact_ot_fm" ;;
    *) echo "Unknown flow matcher: $1" >&2; return 1 ;;
  esac
}

run_dir_for() {
  local matcher="$1"
  local method="$2"
  local nfe="$3"
  local task="$4"
  echo "${OUTPUT_ROOT}/$(matcher_name "${matcher}")/${method}/nfe${nfe}/${task}"
}

eval_seed_for() {
  local dr="$1"
  if [[ "${OFFSET_EVAL_SEED_BY_DR}" == "1" ]]; then
    echo "$((EVAL_SEED + dr))"
  else
    echo "${EVAL_SEED}"
  fi
}

drl_dir() {
  local dr="$1"
  local eval_seed="$2"
  echo "randomization_level_${dr}/scene_${DR_SCENE_MODE}/seed_${eval_seed}"
}

latest_stats_file() {
  local run_dir="$1"
  local policy="$2"
  local task="$3"
  local dr="$4"
  local eval_seed="$5"
  local root="${run_dir}/eval/${task}/${policy}/franka/$(drl_dir "${dr}" "${eval_seed}")"
  if [[ ! -d "${root}" ]]; then
    return 1
  fi
  find "${root}" -name final_stats.txt -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

checkpoint_is_readable() {
  local ckpt="$1"
  [[ -f "${ckpt}" ]] || return 1
  python -c 'import sys, zipfile
path = sys.argv[1]
try:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
    raise SystemExit(1 if bad else 0)
except Exception:
    raise SystemExit(1)
' "${ckpt}" >/dev/null 2>&1
}

for task in "${TASKS[@]}"; do
  zarr="./data_policy/${task}FrankaL${TRAIN_DR_LEVEL}_obs:${OBS_SPACE}_act:${ACT_SPACE}_${DEMO_NUM}.zarr"
  if [[ ! -d "${zarr}" ]]; then
    echo "Missing dataset: ${zarr}" >&2
    echo "Collect/convert ${DEMO_NUM} demos first, or set OBS_SPACE/ACT_SPACE/DEMO_NUM/TRAIN_DR_LEVEL to match your zarr name." >&2
    exit 1
  fi
done

debug_header

for matcher in "${MATCHERS[@]}"; do
  matcher_target_value="$(matcher_target "${matcher}")"
  matcher_label="$(matcher_name "${matcher}")"

  for nfe in "${NFE_STEPS[@]}"; do
    for task in "${TASKS[@]}"; do
      zarr="./data_policy/${task}FrankaL${TRAIN_DR_LEVEL}_obs:${OBS_SPACE}_act:${ACT_SPACE}_${DEMO_NUM}.zarr"

      for method in "${METHODS[@]}"; do
        policy="$(method_policy "${method}")"
        read -r -a method_extra_overrides <<< "$(method_overrides "${method}")"
        run_dir="$(run_dir_for "${matcher}" "${method}" "${nfe}" "${task}")"
        ckpt_path="${run_dir}/checkpoints/${EPOCHS}.ckpt"
        exp_name="${EXP_GROUP}/${matcher_label}/${method}/nfe${nfe}"

        echo ""
        echo "=== train | ${matcher_label} | ${method} (${policy}) | ${task} | NFE=${nfe} | ${EPOCHS} epochs | ${DEMO_NUM} demos | train DR=${TRAIN_DR_LEVEL} ==="
        echo "Flow matcher: ${matcher_target_value}"
        echo "Output: ${run_dir}"
        if [[ "${RUN_TRAIN}" == "1" && "${SKIP_EXISTING_TRAIN}" == "1" ]] && checkpoint_is_readable "${ckpt_path}"; then
          echo "Skip train: existing readable checkpoint ${ckpt_path}"
        elif [[ "${RUN_TRAIN}" == "1" ]]; then
          if [[ -f "${ckpt_path}" ]] && ! checkpoint_is_readable "${ckpt_path}"; then
            echo "Existing checkpoint is missing/corrupt or incomplete; retraining: ${ckpt_path}"
          fi
          export policy_name="${policy}"
          python "${MAIN_SCRIPT}" --config-name=default_runner.yaml \
            "policy_name=${policy}" \
            "task_name=${task}" \
            "exp_name=${exp_name}" \
            "dataset_config.zarr_path=${zarr}" \
            "train_config.training_params.seed=${SEED}" \
            "train_config.training_params.resume=${RESUME_TRAIN}" \
            "train_config.training_params.num_epochs=${EPOCHS}" \
            "train_config.training_params.device=${TRAIN_DEVICE}" \
            "eval_config.policy_runner.obs.obs_type=${OBS_SPACE}" \
            "eval_config.policy_runner.action.action_type=${ACT_SPACE}" \
            "eval_config.eval_args.task=${task}" \
            "eval_config.eval_args.max_step=${EVAL_MAX_STEP}" \
            "eval_config.eval_args.num_envs=${EVAL_NUM_ENVS}" \
            "eval_config.eval_args.sim=${SIM}" \
            "eval_config.eval_args.level=0" \
            "+eval_config.eval_args.gpu_id=${GPU}" \
            "+eval_config.eval_args.max_demo=${MAX_DEMO}" \
            "policy_config.flow_matcher._target_=${matcher_target_value}" \
            "policy_config.flow_matcher.num_sampling_steps=${nfe}" \
            "hydra.run.dir=${run_dir}" \
            "hydra.sweep.dir=${run_dir}" \
            "checkpoint.save_root_dir=${run_dir}" \
            "multi_run.run_dir=${run_dir}" \
            "logging.group=${RUN_TAG}" \
            "logging.name=${matcher_label}_${method}_${task}_nfe${nfe}" \
            "train_enable=True" \
            "eval_enable=False" \
            "${method_extra_overrides[@]}"
        fi

        if [[ "${RUN_EVAL}" == "1" ]]; then
          if ! checkpoint_is_readable "${ckpt_path}"; then
            echo "Missing or unreadable checkpoint: ${ckpt_path}" >&2
            echo "Run training first, or set RUN_TRAIN=1/SKIP_EXISTING_TRAIN=0 to retrain this config." >&2
            exit 1
          fi

          for dr in "${DR_LEVELS[@]}"; do
            eval_seed="$(eval_seed_for "${dr}")"
            eval_dir_hint="${run_dir}/eval/${task}/${policy}/franka/$(drl_dir "${dr}" "${eval_seed}")"
            echo ""
            echo "=== eval | ${matcher_label} | ${method} (${policy}) | ${task} | NFE=${nfe} | randomization_level=${dr} | seed=${eval_seed} | ${EVAL_NUM_ENVS} envs ==="
            echo "Eval output: ${eval_dir_hint}"

            existing_stats_file="$(latest_stats_file "${run_dir}" "${policy}" "${task}" "${dr}" "${eval_seed}" || true)"
            if [[ "${SKIP_EXISTING_EVAL}" == "1" && -n "${existing_stats_file}" && -f "${existing_stats_file}" ]]; then
              success_rate="$(awk -F': ' '/Average Success Rate/ {print $2}' "${existing_stats_file}" | tail -n 1)"
              echo "${RUN_TAG},${method},${policy},${matcher_label},${matcher_target_value},${nfe},${task},${TRAIN_DR_LEVEL},${dr},${DR_SCENE_MODE},${DEMO_NUM},${EPOCHS},${SEED},${eval_seed},${success_rate},${existing_stats_file}" >> "${RESULTS_CSV}"
              echo "Skip eval: existing stats ${existing_stats_file}"
              continue
            fi

            export policy_name="${policy}"
            python "${MAIN_SCRIPT}" --config-name=default_runner.yaml \
              "policy_name=${policy}" \
              "task_name=${task}" \
              "exp_name=${exp_name}" \
              "dataset_config.zarr_path=${zarr}" \
              "train_config.training_params.seed=${SEED}" \
              "train_config.training_params.num_epochs=${EPOCHS}" \
              "train_config.training_params.device=${TRAIN_DEVICE}" \
              "eval_config.policy_runner.obs.obs_type=${OBS_SPACE}" \
              "eval_config.policy_runner.action.action_type=${ACT_SPACE}" \
              "eval_config.eval_args.task=${task}" \
              "eval_config.eval_args.max_step=${EVAL_MAX_STEP}" \
              "eval_config.eval_args.num_envs=${EVAL_NUM_ENVS}" \
              "eval_config.eval_args.sim=${SIM}" \
              "eval_config.eval_args.level=${dr}" \
              "+eval_config.eval_args.scene_mode=${DR_SCENE_MODE}" \
              "+eval_config.eval_args.randomization_seed=${eval_seed}" \
              "+eval_config.eval_args.gpu_id=${GPU}" \
              "+eval_config.eval_args.max_demo=${MAX_DEMO}" \
              "policy_config.flow_matcher._target_=${matcher_target_value}" \
              "policy_config.flow_matcher.num_sampling_steps=${nfe}" \
              "hydra.run.dir=${run_dir}" \
              "hydra.sweep.dir=${run_dir}" \
              "checkpoint.save_root_dir=${run_dir}" \
              "multi_run.run_dir=${run_dir}" \
              "logging.group=${RUN_TAG}" \
              "logging.name=${matcher_label}_${method}_${task}_nfe${nfe}_dr${dr}" \
              "train_enable=False" \
              "eval_enable=True" \
              "eval_path=${ckpt_path}" \
              "${method_extra_overrides[@]}"

            stats_file="$(latest_stats_file "${run_dir}" "${policy}" "${task}" "${dr}" "${eval_seed}" || true)"
            if [[ -n "${stats_file}" && -f "${stats_file}" ]]; then
              success_rate="$(awk -F': ' '/Average Success Rate/ {print $2}' "${stats_file}" | tail -n 1)"
              echo "${RUN_TAG},${method},${policy},${matcher_label},${matcher_target_value},${nfe},${task},${TRAIN_DR_LEVEL},${dr},${DR_SCENE_MODE},${DEMO_NUM},${EPOCHS},${SEED},${eval_seed},${success_rate},${stats_file}" >> "${RESULTS_CSV}"
              echo "Recorded ${matcher_label}, ${method}, ${task}, NFE=${nfe}, randomization_level=${dr}: ${success_rate}"
            else
              echo "${RUN_TAG},${method},${policy},${matcher_label},${matcher_target_value},${nfe},${task},${TRAIN_DR_LEVEL},${dr},${DR_SCENE_MODE},${DEMO_NUM},${EPOCHS},${SEED},${eval_seed},NA,NA" >> "${RESULTS_CSV}"
              echo "Warning: final_stats.txt not found for ${matcher_label}, ${method}, ${task}, NFE=${nfe}, randomization_level=${dr}" >&2
            fi
          done
        fi
      done
    done
  done
done

echo ""
echo "Summary written to ${RESULTS_CSV}"
echo "Manifest written to ${MANIFEST}"
