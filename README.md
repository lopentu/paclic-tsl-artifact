# Naming Is Not Locating — code and data

Code, annotations and model outputs for

> Kai-Yang Peng, Pin-Er Chen and Shu-Kai Hsieh. *Naming Is Not Locating: A Cue-Ablation
> Case Study of Metalinguistic Judgments about Taiwan Sign Language in Two Vision–Language
> Models.* PACLIC 2026.

The paper asks a vision–language model for a form description and five functional ratings of
55 target actions in Taiwan Sign Language discourse, and removes the translation and the
articulator cue in a 2×2 design. Qwen3-VL-8B-Instruct receives the full set of conditions,
GLM-4.6V-Flash the core tests, and Gemma 4 E4B and MiMo-VL-7B the cue ablation without human
rating. This repository holds what every number in the paper is computed from.

## Reproducing the paper's numbers

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-core.txt
PYTHON=.venv/bin/python bash reproduce.sh
git status --short data/results
```

`reproduce.sh` reruns every scoring and analysis step on the released model outputs and
annotations, and overwrites each table and report in `data/results/` with the regenerated one.
No video and no GPU are needed. On an unmodified checkout `git status` lists two files, and
in each the only change is four lines: the frame counts at the end of the two
`length_and_tie_checks.md` reports are measured from the cut video clips, and without the clips
the script prints a note in their place. Everything else regenerates byte-identically.

`results/qwen3vl_8b/metalinguistic_probe.md` is not rewritten, because
`scripts/20_run_metalinguistic_probe.py` writes that transcript only alongside inference.

## What is here

| Path | Contents |
|---|---|
| `reproduce.sh` | The analysis chain, from released outputs to every report |
| `scripts/` | The pipeline: fetching and cutting clips, inference, scoring, analysis |
| `scripts/config/prompts/` | The prompt for all four cue cells, its two inventory variants, the anchored-rubric and mediation prompts, and the JSON schema every output is validated against |
| `scripts/config/pipeline_config.*.yaml` | Frame rate, pixel budget and decoding settings, one file per model and prompt variant |
| `clip_manifest.csv` | The 40 analysis windows and 55 target events: start and end times, target articulator, signer layout |
| `data/sources.csv` | `video_id` → YouTube URL for the 11 source videos |
| `data/clip_manifest_crossmodel_pilot.csv` | The 20 events of the format-only screen used to choose the second model |
| `data/reference_annotations/` | The two annotators' 0–3 ratings (`events_checked.csv`), the Tier-1 ratings of both human-rated models with their rubrics, the uncapped re-scores, and the handedness labels |
| `data/model_outputs/` | Every model run reported in the paper, one JSON file per run |
| `data/results/` | The scored tables and analysis reports the paper's numbers are read from |

`data/model_outputs/` is organized by model:

- `qwen3vl_8b/` — the four cue cells, both as the main runs (`__combined__` in the file name:
  the window clip and the target-span clip in one call) and as the single-clip variant
  (`__ev1__`/`__ev2__`: the target-span clip alone), and one subdirectory per further condition:
  `visual_null/` (degraded video), `temperature_t1/` (*T* = 1.0, five seeded draws per item in
  two cells), `miscue/` and `miscue_resting/` (the two mis-cue schemes), `mediation/` (the
  description handed back as text), `wrongclip/`, `anchored/`, `metalinguistic_probe/` and
  `handedness_probe/`;
- `qwen3vl_8b_altlist/`, `qwen3vl_8b_exclusive/` — the two inventory variants, in the two
  articulator-withheld cells;
- `glm46v_flash/` — the four cue cells and both mis-cue schemes;
- `gemma4_e4b/`, `mimo_vl_7b/` — the four cue cells.

Each output records the cue cell, the clip, the model's raw text, the parsed JSON and its schema
verdict. For GLM-4.6V-Flash and MiMo-VL-7B, `scripts/35_normalize_type_only_errors.py` has
admitted the few outputs whose only schema violations are JSON types the prompt never
specifies; each such file keeps the strict verdict in `schema_valid_strict` and
`schema_errors_strict` and names the converted fields in `post_hoc_normalization`, and
`raw_text` is untouched.

## Running the models

Inference needs the source videos, `ffmpeg`, and a 24 GB GPU.

```bash
.venv/bin/pip install -r requirements-model.txt
python scripts/01_fetch_source_videos.py --url-list data/sources.csv   # to data/raw/
python scripts/03b_cut_event_clips.py                 # analysis windows and target-span clips
python scripts/04_deidentify_subtitles.py --source clip_manifest   # blur captions and overlays
python scripts/07b_run_inference_events.py --mode grounded --clip-kind combined
# ... and --mode ablate_translation, ablate_channel, ablate_both
# single-clip variant: --clip-kind ev1, then --clip-kind ev2
```

Other models take `--config scripts/config/pipeline_config.<model>.yaml`. The follow-up
conditions have their own runners: `19` (degraded video), `20` (text-only probes), `21`
(handedness), `26` (mis-cues), `27` (mediation), `28` (*T* = 1.0), `31` (wrong clip) and
`32` (anchored rubric). Every scored run uses greedy decoding with one run per item, so the
same checkpoint and library versions should reproduce those outputs; the *T* = 1.0 draws are
seeded per (clip, event, cell, draw) but are not bit-reproducible across library or hardware
versions.

**The translation cue.** `clip_manifest.csv` keeps a `translation` column, but it is empty: the
translation given to the model in the translation-given cells is the program's own Chinese
subtitle for the window, and that text is not redistributed. To rerun those cells, fill the
column with the subtitle shown during each window. The runners stop with an error if a
translation-given cell meets an empty column. The articulator-only and no-cue cells need
nothing.

## What is not here

**No video and no frames.** Fetch the videos from `data/sources.csv` and cut them with the
scripts above; every timestamp is in `clip_manifest.csv`.

**No subtitle text.** Besides the empty manifest column, the two places where an output record
copied its input prompt are redacted: the `prompt` field of `handedness_probe/` and the
`prompt_from.translation` field of `wrongclip/`, where each subtitle line is replaced by
`<translation not redistributed>`. The model's own text (`raw_text`, `parsed`, and the
descriptions handed back in `mediation/`) is published as generated, and sometimes quotes the
translation it was given.

**No source titles or local paths.** Videos are identified by ID and URL only, and absolute
paths in the outputs are written as `<repo>`.

**Nothing outside the reported experiment.** Only the clip formats, conditions and models
that the paper reports are included.

## Which file backs which claim

Section, table and figure numbers refer to the published paper.

| Paper | Report in `data/results/` | Produced by |
|---|---|---|
| §2 reference agreement and its per-articulator structure | `qwen3vl_8b/annotator_agreement_checked.md`, `qwen3vl_8b/label_priors.md` | `12_compute_annotator_agreement.py`, `24_analyze_label_priors.py` |
| §3 frames per item | `qwen3vl_8b/length_and_tie_checks.md` (needs the clips) | `25_analyze_length_and_tie_checks.py` |
| §3, Appendix C model screening | `crossmodel_pilot_screen.md` | `34_screen_crossmodel_pilot.py` |
| Table 2, §4.1 adequacy, McNemar tests and bootstrap intervals | `qwen3vl_8b/robustness_checks_combined.md`, `qwen3vl_8b/perception_agreement_events_combined.md`; GLM and the between-model contrast in `glm46v_flash/human_rated_crossmodel_combined.md` | `09b`, `12b`, `18`, `36` |
| §4.1, Appendix A uncapped re-score | `qwen3vl_8b/uncapped_rescore_combined.md`; GLM in `glm46v_flash/human_rated_crossmodel_combined.md` | `23_fill_uncapped_scores.py`, `36` |
| Table 5, adequacy by target articulator | `qwen3vl_8b/perception_agreement_events_combined.md`; GLM in `glm46v_flash/human_rated_crossmodel_combined.md` | `12b`, `36` |
| §4.2 self-selection and the inventory variants | `qwen3vl_8b_altlist/` and `qwen3vl_8b_exclusive/prompt_variant_channel_comparison.md`, `qwen3vl_8b/temperature_stability.md` | `compare_prompt_variant_channel.py`, `30` |
| §4.2, Appendix C relabeling test, containment, permutation null and intervals | `qwen3vl_8b/channel_relabeling_combined.md`, `qwen3vl_8b/length_and_tie_checks.md` | `analyze_channel_relabeling.py`, `25` |
| Table 3, §4.3 Tier 2, `art.` and `det.` | `qwen3vl_8b/attribution_summary_combined_checked.md`, `qwen3vl_8b/robustness_checks_combined.md`; GLM in `glm46v_flash/human_rated_crossmodel_combined.md` | `10b`, `analyze_attribution_summary.py`, `18`, `36` |
| Table 6, articulator means and model modes | `qwen3vl_8b/label_priors.md`; GLM in `glm46v_flash/human_rated_crossmodel_combined.md` | `24`, `36` |
| Table 7, all model–reference correlations with BH adjustment | `qwen3vl_8b/robustness_checks_combined.md`; GLM in `glm46v_flash/human_rated_crossmodel_combined.md` | `18`, `36` |
| §4.4 mis-cues and the mediation probe | `qwen3vl_8b/miscue_and_mediation.md`, `qwen3vl_8b/followup_conditions.md`; GLM in `glm46v_flash/miscue_and_mediation.md`, `glm46v_flash/followup_conditions.md` | `29_analyze_miscue_and_mediation.py`, `33_analyze_followup_conditions.py` |
| Table 4, §4.5 degraded video and the label-only row | `qwen3vl_8b/visual_null.md`, `qwen3vl_8b/label_priors.md` | `analyze_visual_null.py`, `24` |
| §4.5 wrong clip; Appendix C ladder extension and rubric anchoring | `qwen3vl_8b/followup_conditions.md` | `33` |
| Table 8, the four models | `crossmodel/cross_model_all4_combined.md` | `analyze_cross_model.py` |
| Appendix B text-only and handedness probes | `qwen3vl_8b/metalinguistic_probe.md`, `qwen3vl_8b/handedness_probe.md` | `20`, `21` |
| Appendix C lexical-sign appeals, fixed-denominator and gating checks, depiction | `qwen3vl_8b/robustness_checks_combined.md`, `qwen3vl_8b/label_priors.md` | `18`, `24` |
| Appendix C single-clip variant | `qwen3vl_8b/single_clip_event.md` | `analyze_cross_model.py --clip-kind event` |
| Appendix C sampling at *T* = 1.0 | `qwen3vl_8b/temperature_stability.md` | `30_analyze_temperature_stability.py` |
| Appendix C Tier-1 agreement, Gwet's AC1/AC2 | `qwen3vl_8b/perception_agreement_events_combined.md`, `qwen3vl_8b/length_and_tie_checks.md`; GLM in `glm46v_flash/human_rated_crossmodel_combined.md` | `12b`, `25`, `36` |

## Ethics

Only publicly available video is used. The videos are not redistributed, and the repository
carries timestamps, annotations and model outputs only. Captions and channel-identifying
overlays are blurred before any frame reaches a model. The two annotators appear as `A01` and
`A02`. The results describe model behavior on a small sample from one program, not signers or
their language.

## Licence

Code (`scripts/`, `reproduce.sh`) is released under the MIT Licence (`LICENSE`). The model
outputs, annotations, manifests and reports in `data/` and `clip_manifest.csv` are released
under CC BY 4.0 (`LICENSE-DATA`). The source videos, their subtitles and the model checkpoints
are governed by their own terms and are covered by neither.
