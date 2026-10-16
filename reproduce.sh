#!/usr/bin/env bash
# Regenerate the scored tables and analysis reports in data/results/ from the released model
# outputs and annotations. No video and no GPU: install requirements-core.txt only.
#
#   PYTHON=.venv/bin/python bash reproduce.sh
#   git status --short data/results
#
# Each step overwrites the file it produces. On an unmodified checkout the only change is the
# frame-count section of the two length_and_tie_checks.md reports, which is measured from the cut
# video clips and is replaced by a note when they are absent (see README).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
REF=data/reference_annotations/events_checked.csv
QWEN=scripts/config/pipeline_config.qwen3vl_8b.yaml
GLM=scripts/config/pipeline_config.glm46v_flash.yaml

run() { echo "+ $*"; "$PY" "$@" > /dev/null; }

# --- Qwen3-VL-8B-Instruct: the four cue cells --------------------------------------------------
for m in grounded ablate_translation ablate_channel ablate_both; do
  run scripts/08b_score_localization_events.py --config "$QWEN" --mode "$m"
  run scripts/09b_score_perception_adequacy_events.py aggregate --config "$QWEN" --mode "$m" \
      --clip-kind combined --filled-dir "data/reference_annotations/perception_filled_combined_$m"
  run scripts/10b_score_attribution_events.py --config "$QWEN" --mode "$m" --clip-kind combined \
      --events-csv "$REF" --out-suffix _checked
  run scripts/10b_score_attribution_events.py --config "$QWEN" --mode "$m" --clip-kind combined \
      --events-csv "$REF" --ignore-perception-gate
done
run scripts/12_compute_annotator_agreement.py --config "$QWEN" --events-csv "$REF" --out-suffix _checked
run scripts/12b_compute_perception_agreement_events.py --config "$QWEN"
run scripts/analyze_attribution_summary.py --config "$QWEN" --suffix _checked
run scripts/analyze_channel_relabeling.py --config "$QWEN"
run scripts/18_analyze_robustness_checks.py --config "$QWEN"
run scripts/23_fill_uncapped_scores.py report --config "$QWEN"
run scripts/24_analyze_label_priors.py --model qwen3vl_8b
run scripts/25_analyze_length_and_tie_checks.py --config "$QWEN"
run scripts/analyze_visual_null.py --config "$QWEN"
run scripts/29_analyze_miscue_and_mediation.py --config "$QWEN"
run scripts/30_analyze_temperature_stability.py --k 5
run scripts/33_analyze_followup_conditions.py --config "$QWEN"
run scripts/21_run_handedness_probe.py --config "$QWEN" --report --truth-csv data/reference_annotations/handedness_truth.csv
run scripts/compare_prompt_variant_channel.py --variant-tag qwen3vl_8b_altlist
run scripts/compare_prompt_variant_channel.py --variant-tag qwen3vl_8b_exclusive
run scripts/analyze_cross_model.py --models qwen3vl_8b --clip-kind event \
    --out data/results/qwen3vl_8b/single_clip_event.md

# --- GLM-4.6V-Flash: the two human-rated cells and both mis-cues -------------------------------
run scripts/36_analyze_crossmodel_human_rated.py split
for m in grounded ablate_channel; do
  run scripts/08b_score_localization_events.py --config "$GLM" --mode "$m"
  run scripts/09b_score_perception_adequacy_events.py aggregate --config "$GLM" --mode "$m" \
      --clip-kind combined --filled-dir "data/reference_annotations/perception_round2/filled_glm46v_flash/$m"
  run scripts/10b_score_attribution_events.py --config "$GLM" --mode "$m" --clip-kind combined \
      --events-csv "$REF" --out-suffix _checked
  run scripts/10b_score_attribution_events.py --config "$GLM" --mode "$m" --clip-kind combined \
      --events-csv "$REF" --ignore-perception-gate
done
run scripts/12b_compute_perception_agreement_events.py --config "$GLM"
run scripts/analyze_attribution_summary.py --config "$GLM" --suffix _checked
run scripts/25_analyze_length_and_tie_checks.py --config "$GLM"
run scripts/29_analyze_miscue_and_mediation.py --config "$GLM" --skip-mediation
run scripts/33_analyze_followup_conditions.py --config "$GLM" --skip-wrongclip --skip-anchored
run scripts/36_analyze_crossmodel_human_rated.py report

# --- All four models: the format screen and the reference-free comparison ---------------------
run scripts/34_screen_crossmodel_pilot.py
run scripts/analyze_cross_model.py --models qwen3vl_8b glm46v_flash gemma4_e4b mimo_vl_7b \
    --clip-kind combined --out data/results/crossmodel/cross_model_all4_combined.md

echo "done: compare with git status / git diff data/results"
