# Cross-model pilot: format screen

Rule (the project design log 2026-10-09): all 40 calls complete and >= 38/40 schema-valid; first passing model in the fixed order is primary. Format statistics only.

| order | model tag | completed | schema-valid | reasoning leaked | empty think tags | input tokens (min-max) | output tokens (max) | pass |
|---|---|---|---|---|---|---|---|---|
| 1 | glm46v_flash | 40/40 | 39/40 | 0 | 0 | 7022-16204 | 347 | PASS |
| 2 | gemma4_e4b | 40/40 | 40/40 | 0 | 0 | 6067-13484 | 321 | PASS |
| 3 | mimo_vl_7b | 40/40 | 38/40 | 0 | 40 | n/a | n/a | PASS |

**Primary replication model: glm46v_flash**

- glm46v_flash: both videos fed on 40/40 logged calls; window frames 42-136, target frames 12-40
- gemma4_e4b: both videos fed on 40/40 logged calls; window frames 42-136, target frames 12-40

### glm46v_flash failures

- C08/event1/ablate_channel: candidate_actions/0/end_time: 0.8 is not of type 'string'; candidate_actions/0/start_time: 0.3 is not of type 'string'

### mimo_vl_7b failures

- C07/event1/ablate_channel: candidate_actions/0/depiction: '0' is not of type 'integer'; candidate_actions/0/description: '2' is not of type 'integer'; candidate_actions/0/indication: '1' 
- C16/event2/ablate_channel: candidate_actions/0/depiction: '1' is not of type 'integer'; candidate_actions/0/description: '1' is not of type 'integer'; candidate_actions/0/indication: '1' 
