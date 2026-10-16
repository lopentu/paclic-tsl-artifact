# Prompt M — description → D1 mediation probe

Text-only. No video is sent. The model is handed back a form description that it produced
itself, on an earlier run, from the real clip, and is asked for the same functional scores it
gave then.

Two arms, differing in ONE sentence:

    LABEL_WITHHELD   the description alone
    LABEL_GIVEN      the description plus an explicit "carried by __TARGET_CHANNEL__" line

Everything else — the scale definitions, the output format, the wording — is held byte-identical
between the arms and is copied verbatim from Prompt C's rating section, so a score difference
cannot be a difference in how the question was asked.

---

You are analyzing a description of a single action from a Taiwan Sign Language utterance. You
are NOT being shown video. Someone who watched the clip has written down what the signer did,
and your task is to judge that action's linguistic function from their description alone.

[[MEANING_GIVEN]]
The utterance this action occurs in means: __UTTERANCE_MEANING__
[[/MEANING_GIVEN]]

[[LABEL_GIVEN]]
The action is carried by __TARGET_CHANNEL__.
[[/LABEL_GIVEN]]

Description of the action:

__FORM_DESCRIPTION__

Rate that action on the following, using the same 0–3 scales:

- `semantic_syntactic_load` — how much the action contributes to the utterance's propositional
  content or grammatical structure. 0 = no clear contribution; 1 = minor or redundant
  contribution; 2 = a substantive contribution that the utterance would be poorer without;
  3 = removing the action changes core meaning, grammatical relations, referent tracking, or
  discourse structure.
- `pragmatic_depictive_affective_enrichment` — how much the action adds stance, emphasis,
  vividness, affect, enactment or depiction. 0 = none; 3 = strong enrichment.
- `description` — conventionalized encoded meaning (0–3).
- `indication` — pointing, locating, indexing (0–3).
- `depiction` — showing or enacting (0–3).

These are independent: an action can score high on more than one.

Answer with a single JSON object and nothing else:

```json
{
  "semantic_syntactic_load": <int 0-3>,
  "pragmatic_depictive_affective_enrichment": <int 0-3>,
  "description": <int 0-3>,
  "indication": <int 0-3>,
  "depiction": <int 0-3>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "one or two sentences"
}
```

---

## NOTE ON USE (not sent to the model)

`__FORM_DESCRIPTION__` is filled with the model's own `form_description` fields from the
corresponding `grounded` run, rendered as `field: value` lines, with empty and
`not applicable` / `not visible` fields dropped.

The `hands` field is dropped from BOTH arms. Its value is "dominant hand" / "non-dominant hand" /
"not visible" — it states the articulator rather than describing the action, so keeping it would
put the label back into the arm that is defined by not having one. Same exclusion, and same
reason, as the description-similarity measure's dropping of `hand(s)` / `signer`.

The point of the probe is to separate two paths to a D1 score that the no-video condition
cannot: in that condition the prompt still carries the articulator label, so a D1 that survives
losing the video could be following either the label or the model's own description of what it
saw. Here the two are dissociated by arm.

    D1(LABEL_WITHHELD)  == D1(video run)   ->  the score is a function of the DESCRIPTION
    D1(LABEL_WITHHELD)  != D1(video run)
      but D1(LABEL_GIVEN) == D1(video run) ->  the score is a function of the LABEL

## NOTE ON THE OUTPUT EXAMPLE

The output example above shows every numeric field as a TYPE PLACEHOLDER (`<int 0-3>`,
`<float 0.0-1.0>`), never as a filled-in value. A concrete numeral in an example reads to the
model as an answer and is liable to be copied verbatim; a placeholder reads as an instruction to
supply one. Do not "helpfully" fill these in when editing this prompt.
