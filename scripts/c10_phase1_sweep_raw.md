# C.10 Phase 1 — Merged-chunk calibration sweep
Bundle: `air_defense_v3`. Docs: `Dvina`, `SA-2`.
Sweep grid: min_similarity=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], top_n_candidates=[25, 50, 75, 100], top_k=[5, 10, 15, 20, 30] → 140 cells/(doc,pass).
Ground truth: top-15 chunks by full-corpus reranker score (max-permissive).
Knee criterion: smallest config (token-then-chunk-count) reaching ≥95% of max coverage.

## Per-pass knee selection

| doc | pass | min_sim | top_n | top_k | sel_chunks | exp_refs | sel_tokens | cov | max_cov |
|---|---|---|---|---|---|---|---|---|---|
| Dvina | radar_power_rf | 0.20 | 50 | 15 | 15 | 10 | 7288 | 100% | 100% |
| Dvina | radar_antenna | 0.20 | 50 | 15 | 15 | 8 | 7680 | 100% | 100% |
| Dvina | radar_timing | 0.20 | 50 | 15 | 15 | 16 | 4386 | 100% | 100% |
| Dvina | radar_modulation | 0.20 | 50 | 15 | 15 | 28 | 7433 | 100% | 100% |
| Dvina | missile_kinematics | 0.20 | 50 | 15 | 15 | 8 | 7680 | 100% | 100% |
| Dvina | missile_guidance | 0.20 | 50 | 15 | 15 | 8 | 6184 | 100% | 100% |
| Dvina | missile_airframe | 0.20 | 50 | 15 | 15 | 28 | 7185 | 100% | 100% |
| Dvina | missile_speed_timing | 0.20 | 50 | 15 | 15 | 25 | 6785 | 100% | 100% |
| Dvina | missile_propulsion | 0.20 | 50 | 15 | 15 | 8 | 4883 | 100% | 100% |
| SA-2 | radar_power_rf | 0.20 | 50 | 15 | 15 | 39 | 5741 | 100% | 100% |
| SA-2 | radar_antenna | 0.20 | 50 | 15 | 15 | 39 | 6429 | 100% | 100% |
| SA-2 | radar_timing | 0.20 | 50 | 15 | 15 | 40 | 5460 | 100% | 100% |
| SA-2 | radar_modulation | 0.20 | 50 | 15 | 15 | 39 | 6580 | 100% | 100% |
| SA-2 | missile_kinematics | 0.20 | 50 | 15 | 15 | 45 | 5617 | 100% | 100% |
| SA-2 | missile_guidance | 0.20 | 50 | 15 | 15 | 42 | 5751 | 100% | 100% |
| SA-2 | missile_airframe | 0.20 | 50 | 15 | 15 | 39 | 6163 | 100% | 100% |
| SA-2 | missile_speed_timing | 0.20 | 50 | 15 | 15 | 43 | 6360 | 100% | 100% |
| SA-2 | missile_propulsion | 0.20 | 50 | 15 | 15 | 40 | 6067 | 100% | 100% |

## Per-pass aggregate (mean across docs)

Recommended values are the mean of the per-doc knee.

| pass | min_sim | top_n | top_k | mean_sel_chunks | mean_exp_refs | mean_sel_tokens |
|---|---|---|---|---|---|---|
| radar_power_rf | 0.20 | 50 | 15 | 15.0 | 24.5 | 6514 |
| radar_antenna | 0.20 | 50 | 15 | 15.0 | 23.5 | 7054 |
| radar_timing | 0.20 | 50 | 15 | 15.0 | 28.0 | 4923 |
| radar_modulation | 0.20 | 50 | 15 | 15.0 | 33.5 | 7006 |
| missile_kinematics | 0.20 | 50 | 15 | 15.0 | 26.5 | 6648 |
| missile_guidance | 0.20 | 50 | 15 | 15.0 | 25.0 | 5968 |
| missile_airframe | 0.20 | 50 | 15 | 15.0 | 33.5 | 6674 |
| missile_speed_timing | 0.20 | 50 | 15 | 15.0 | 34.0 | 6572 |
| missile_propulsion | 0.20 | 50 | 15 | 15.0 | 24.0 | 5475 |

## Diagnostic: vector-score distribution

| doc | pass | min_vec | median_vec | max_vec | max_cov |
|---|---|---|---|---|---|
| Dvina | radar_power_rf | (see stdout) | (see stdout) | (see stdout) | 100% |
| Dvina | radar_antenna | (see stdout) | (see stdout) | (see stdout) | 100% |
| Dvina | radar_timing | (see stdout) | (see stdout) | (see stdout) | 100% |
| Dvina | radar_modulation | (see stdout) | (see stdout) | (see stdout) | 100% |
| Dvina | missile_kinematics | (see stdout) | (see stdout) | (see stdout) | 100% |
| Dvina | missile_guidance | (see stdout) | (see stdout) | (see stdout) | 100% |
| Dvina | missile_airframe | (see stdout) | (see stdout) | (see stdout) | 100% |
| Dvina | missile_speed_timing | (see stdout) | (see stdout) | (see stdout) | 100% |
| Dvina | missile_propulsion | (see stdout) | (see stdout) | (see stdout) | 100% |
| SA-2 | radar_power_rf | (see stdout) | (see stdout) | (see stdout) | 100% |
| SA-2 | radar_antenna | (see stdout) | (see stdout) | (see stdout) | 100% |
| SA-2 | radar_timing | (see stdout) | (see stdout) | (see stdout) | 100% |
| SA-2 | radar_modulation | (see stdout) | (see stdout) | (see stdout) | 100% |
| SA-2 | missile_kinematics | (see stdout) | (see stdout) | (see stdout) | 100% |
| SA-2 | missile_guidance | (see stdout) | (see stdout) | (see stdout) | 100% |
| SA-2 | missile_airframe | (see stdout) | (see stdout) | (see stdout) | 100% |
| SA-2 | missile_speed_timing | (see stdout) | (see stdout) | (see stdout) | 100% |
| SA-2 | missile_propulsion | (see stdout) | (see stdout) | (see stdout) | 100% |
