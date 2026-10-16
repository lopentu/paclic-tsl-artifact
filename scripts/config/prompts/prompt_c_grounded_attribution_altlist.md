You are analyzing a short video clip containing Taiwan Sign Language (TSL) discourse. This clip is the full analysis window -- it has NOT been trimmed to isolate a single action, so you will see surrounding signing before and/or after the action you must evaluate.

WHO TO WATCH. Some clips show a single person signing to the camera. Others are interviews or conversations, so TWO people are visible in the frame. In every clip only ONE of them is the main signer: the person actively producing the signed utterance during the target span. Evaluate that person only. The other person may be visible listening, nodding, or briefly reacting, and none of that belongs in your answer -- do not let their movements enter your form description or your scores. If two people are visible, state in your form description which one you took to be the signer (e.g. "the person on the left of the frame").

POINT OF VIEW. Describe every direction from the SIGNER's own body, never from the camera's or the viewer's. "Left", "right", "forward", "back", "inward" and "outward" always mean the signer's left, right, front, back, in and out. So a hand that appears on the right-hand side of the frame, when the signer faces the camera, is on the signer's LEFT. Where it avoids ambiguity, prefer "dominant hand" / "non-dominant hand" over "left hand" / "right hand".

[[MEANING_GIVEN]]
The overall meaning of what is being signed in this window has already been determined for you. It is:

"__UTTERANCE_MEANING__"

You do NOT need to translate the clip yourself, and you should NOT try to re-translate it. Take the meaning above as given. Your job is a different one: to look at ONE specific visible action and judge what that particular action contributes.

Note on this meaning: it comes from the video's subtitle translation, produced for fluent readability rather than as a sign-by-sign gloss. Its wording and word order can diverge from the original signing's structure, so do not assume the target action corresponds to any specific word or phrase in it. Use it as context for what the passage is about, not as a literal gloss of the one action you are evaluating -- judge that action from what you can see.
[[/MEANING_GIVEN]]
[[MEANING_WITHHELD]]
You are NOT given a translation of this utterance, and you should NOT try to produce a full translation of the clip. Your job is to look at ONE specific visible action and judge what that particular action contributes, working from what you can actually see: its visible form, its timing, and how it relates to the signing around it. Where the utterance's meaning matters to your judgment, say what you were able to infer from the visible signing itself, and say plainly when you could not tell.
[[/MEANING_WITHHELD]]

[[CHANNEL_GIVEN]]
The action you must evaluate occurs at approximately __EVENT_START_SEC__ to __EVENT_END_SEC__ seconds into THIS clip, and is carried by __TARGET_CHANNEL__. Evaluate only that action. Do not describe or score any other visible action in the clip. You may use the surrounding footage (before and after the target span) to judge how the target action relates to the rest of the signing, but keep your form description and scores about the target action itself.
[[/CHANNEL_GIVEN]]
[[CHANNEL_WITHHELD]]
The action you must evaluate occurs at approximately __EVENT_START_SEC__ to __EVENT_END_SEC__ seconds into THIS clip. You are NOT told which part of the body carries it. Decide that yourself: watch that span, and evaluate the single most communicatively significant action in it. It is carried by exactly one of these four, and each is equally likely:
- the face
- the head and upper body (torso)
- the non-dominant hand
- the hands (manual)
An articulator can carry the action while barely moving -- for example a non-dominant hand that holds a handshape or a position in place. Do not assume the action must be the largest or fastest movement you see. Evaluate only that one action. Do not describe or score any other visible action in the clip. You may use the surrounding footage (before and after the target span) to judge how the target action relates to the rest of the signing, but keep your form description and scores about the target action itself.
[[/CHANNEL_WITHHELD]]

Do not rely on any visible written subtitles or captions. If any written text is visible and readable in the video frames, report "caption leakage risk: yes".

A visible action can carry linguistic structure and gesture-like enrichment at the same time. Do not force an either/or judgment on the two scores below.

Dimension 1 (D1): Semantic/Syntactic Load
How much does the target action contribute to propositional content, argument/event structure, grammatical marking, referential tracking, or discourse structure?
0 = no clear contribution
1 = weak or uncertain contribution
2 = clear contribution
3 = essential; removing it would change core meaning, a grammatical relation, referent tracking, or discourse structure

Dimension 2 (D2): Pragmatic/Depictive/Affective Enrichment
How much does the target action contribute stance, emphasis, vividness, affect, enactment, or imagistic depiction?
0 = no clear enrichment
1 = weak or uncertain enrichment
2 = clear enrichment
3 = strong; removing it would reduce vividness, stance, affect, or depictive force

D1 and D2 are independent. The same action may score high on both, low on both, or high on one and low on the other.

Work in this order and report each step:

Step 1. Observable form of the target action, from what you can actually see:
- which articulator carries it (face / head and torso / non-dominant hand alone / manual) -- report this as "channel_judged"
- hands involved
- handshape if visible
- movement path and manner
- location
- facial/head/torso activity
- non-dominant hand behavior
Describe only what is visible, and only for the person you identified as the signer. If the target span shows little or no activity, say so plainly rather than inventing detail.

Step 2. Temporal relation / scope: how the target action's timing relates to the surrounding signing (does it start/end with a particular sign or stretch of signing, span a longer unit, or stand alone in a pause?).

Step 3. Assign D1 (0-3).

Step 4. Assign D2 (0-3).

Step 5. Semiotic mode. Independently of the two scores above, rate HOW the target action conveys its meaning, on three modes that can co-occur (an action may be high on more than one). Each is scored 0-3, on the same scale as D1 and D2: 0 = not this, 1 = partly or weakly this, 2 = clearly this, 3 = strongly this.
- description (0-3): uses a conventionalized linguistic form that carries encoded meaning -- a lexical sign (a "word"), or a conventional grammatical marker.
- indication (0-3): points at, locates, or indexes something/someone in space -- deixis, referent tracking. Note: an action can LOOK like a point (e.g. an extended finger) yet be a lexical/descriptive sign, not indication -- judge the function, not just the handshape.
- depiction (0-3): shows or enacts -- a classifier/depicting handshape standing for an entity and showing its location/motion/manner, or constructed action (enacting a character).
If the action is a purely prosodic or affective nonmanual (e.g. an emphatic or emotional facial expression) or the articulator is simply at rest, all three may be 0 -- its contribution is then captured by the D1/D2 scores above, not by these modes.

Step 6. Observable evidence: one or two short sentences pointing to what in the video supports your D1, D2, and mode ratings. Do not speculate beyond the visible evidence. If the given meaning and the visible evidence point in different directions, say so.

Step 7. Confidence: 0.0 to 1.0.

Respond with ONLY a single JSON object, no prose before or after it, containing exactly one entry in "candidate_actions" (the target action), matching exactly this structure. Report your best estimate of the target action's start/end time within the clip (it should be close to the span given above):

{
  "caption_leakage_risk": "yes/no/uncertain",
  "candidate_actions": [
    {
      "start_time": "approximate timestamp",
      "end_time": "approximate timestamp",
      "channel_judged": "one of: face | head_torso | non_dominant_hand | manual",
      "form_description": {
        "hands": "...",
        "handshape": "...",
        "movement": "...",
        "location": "...",
        "nonmanual": "...",
        "non_dominant_hand": "...",
        "temporal_alignment": "..."
      },
      "semantic_syntactic_load": 0,
      "pragmatic_depictive_affective_enrichment": 0,
      "description": 0,
      "indication": 0,
      "depiction": 0,
      "confidence": 0.0,
      "observable_evidence": "one or two short sentences"
    }
  ]
}

NOTE ON USE: researcher-facing. lib/prompt_utils.load_prompt_template cuts the prompt at
this line, so nothing below it is sent to the model.

Inventory variant. Only the CHANNEL_WITHHELD block differs from
prompt_c_grounded_attribution.md, so the articulator-given cells would render identically;
only ablate_channel and ablate_both are run, with 07b_run_inference_events.py --prompt-file
and --config scripts/config/pipeline_config.qwen3vl_8b_altlist.yaml. The four articulator
options become a parallel bulleted list stated to be equally likely, with manual moved
last, plus one sentence noting that an articulator can carry the action while barely
moving.
