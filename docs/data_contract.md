# Data contract

Source: `reports/batch_failures.json` and `reports/outcome_thresholds.json` as they sit in the repository. Fields that are not in those files are not listed here.

An episode is read from `outcomes` records. All 10200 outcomes in `parameter_sweeps` and all 10200 outcomes in `parameter_sweeps_by_scenario.moderate` share the same set of keys. The first two episodes of level `com_x` / `zero` in scenario `nominal` have no failure. The failure example comes from `floor_contact` / `soft` in the same scenario.

## `reports/batch_failures.json`

The root is an object with 39 fields.

| Field | Type | What it holds |
| --- | --- | --- |
| `n_episodes` | integer | `80` |
| `n_envs` | integer | `80` |
| `n_failures` | integer | `0` |
| `n_survived` | integer | `80` |
| `failure_rate` | number | `0.0` |
| `mode` | string | `baseline` |
| `seed` | integer | `42` |
| `elapsed_s` | number | run duration in seconds |
| `n_workers` | integer | `8` |
| `n_chunks` | integer | `8` |
| `kp` | number | `40.0` |
| `kd` | number | `3.0` |
| `floor_geom_id` | integer | `0` |
| `floor_geom_name` | string | `floor` |
| `floor_geom_type` | string | `plane` |
| `geom_friction_shape` | array of integer | `[57, 3]` |
| `friction_geom_ids` | array of integer | `[0, 20, 32, 44, 56]` |
| `torso_body_id` | integer | `1` |
| `impulse_start` | integer | `450` |
| `impulse_end` | integer | `500` |
| `target_qpos` | array of number | 12 numbers |
| `failed` | object | summary of failed episodes, see below |
| `survived` | object | summary of survived episodes, see below |
| `failures` | array | empty array, no elements |
| `parameter_sweeps` | object | OFAT scenario `nominal` |
| `run_mode` | string | `ofat` |
| `pd` | string | `baseline` |
| `scenario` | string | `default` |
| `scenario_label` | string | `базовый: трение и боковой удар (Fy до ±350 Н)` |
| `friction_range` | array of number | `[0.05, 1.0]` |
| `lateral_force_range` | array of number | `[-350.0, 350.0]` |
| `stance_hold` | object | `friction`, `lateral_force`, `n_episodes`, `n_failures`, `failure_rate` |
| `baseline_grid_path` | string | `reports\baseline_grid.csv` |
| `baseline_grid` | array of object | 42 grid rows, not OFAT episodes |
| `created_at` | string | `2026-11-22T18:09:35.278393+00:00` |
| `config` | object | configuration snapshot, see below |
| `config_hash` | string | `f920c32514104f9b` |
| `parameter_sweeps_by_scenario` | object | keys `nominal` and `moderate` |
| `parameter_sweeps_moderate` | object | the same scenario as `parameter_sweeps_by_scenario.moderate` |

There is no episode array at the root. Per-episode records live here:

`parameter_sweeps.parameters[].levels[].outcomes[]`

The same shape is used by `parameter_sweeps_by_scenario.nominal`, `parameter_sweeps_by_scenario.moderate`, and `parameter_sweeps_moderate`.

### Summaries `failed` and `survived`

Both objects have two keys: `friction` and `lateral_force`. Each of those summaries has four fields.

| Field | Type |
| --- | --- |
| `mean` | number or null |
| `min` | number or null |
| `max` | number or null |
| `count` | integer |

In the current file, every `mean` / `min` / `max` under `failed` is null and `count` is 0. Under `survived` the numbers are filled in and `count` is 80.

### `config`

| Field | Type | Value in the file |
| --- | --- | --- |
| `model_path` | string | path to `unitree_go2\scene.xml` |
| `seed` | integer | `42` |
| `kp` | number | `40.0` |
| `kd` | number | `3.0` |
| `fall_height` | number | `0.2` |
| `contact_z` | number | `0.12` |
| `sag_z` | number | `0.2` |
| `tilt_fail_deg` | number | `45.0` |
| `upright_hold_s` | number | `0.2` |
| `joint_jitter_rad` | number | `0.02` |
| `impulse_jitter_steps` | integer | `10` |
| `mass_jitter` | number | `0.03` |
| `max_steps` | integer | `1500` |
| `impulse_start` | integer | `450` |
| `impulse_end` | integer | `500` |
| `n_envs` | integer | `80` |
| `n_sweep_episodes` | integer | `50` |
| `n_grid_episodes` | integer | `100` |
| `chunk_size` | integer | `10` |
| `scenario` | string | `default` |
| `friction_range` | array of number | `[0.05, 1.0]` |
| `lateral_force_range` | array of number | `[-350.0, 350.0]` |
| `stance_ref_z` | number | `0.2440828391927459` |
| `n_episodes` | integer | `300` |
| `failure_criterion` | string | `fall OR stance_loss` |
| `stance_loss_frac` | number | `0.85` |

### Factor level that contains an episode

Path: `parameter_sweeps.parameters[]` is an object with fields `id`, `name`, `baseline`, `levels`.

One `levels[]` element:

| Field | Type | Note from the file |
| --- | --- | --- |
| `id` | string | for example `zero` |
| `label` | string | for example `базовый 0 м` |
| `factor` | object | the key set depends on the factor |
| `n_episodes` | integer | `300` |
| `n_failures` | integer | |
| `n_survived` | integer | |
| `failure_rate` | number | |
| `failure_rate_ci95` | array of number, length 2 | |
| `n_sag` | integer | |
| `sag_rate` | number | |
| `sag_rate_ci95` | array of number, length 2 | |
| `fail_reasons` | object | reason counts; a zero level can be `{}` |
| `n_tilt` | integer | |
| `tilt_rate` | number | |
| `tilt_rate_ci95` | array of number, length 2 | |
| `n_stance_loss` | integer | |
| `stance_loss_rate` | number | |
| `stance_loss_rate_ci95` | array of number, length 2 | |
| `n_before_impact` | integer | |
| `n_at_or_after_impact` | integer | |
| `min_z_mean` | number | |
| `max_abs_roll_mean` | number | |
| `elapsed_s` | number | |
| `n_workers` | integer | |
| `n_chunks` | integer | |
| `outcomes` | array of object | episodes, see the next section |
| `delta_vs_baseline` | number | |
| `mcnemar` | object or null | null on the baseline level `com_x` / `zero` |
| `p_mcnemar` | number or null | |
| `p_holm` | number or null | |
| `significant` | boolean | |
| `effect_label` | string | the file contains `норма`, `в пределах шума`, `значимо` |

`fail_reasons` is not an episode field. It is a level object. The file contains an empty object, `{"stance_loss": 300}`, and `{"fall": 72, "stance_loss": 61}`.

When `mcnemar` is an object, its keys are: `test` (string, value `mcnemar_exact`), `n_paired`, `n00`, `n01`, `n10`, `n11`, `n_discordant` (integer), `pvalue` (number).

### One episode (`outcomes[]`)

| Field | Type | Example without a failure (`nominal`, `com_x` / `zero`, episodes 1 and 2) | Example with a failure (`nominal`, `floor_contact` / `soft`, episode 1) |
| --- | --- | --- | --- |
| `episode` | integer | `1`, then `2` | `1` |
| `seed` | integer | `42000127`, then `42000128` | `42000127` |
| `failed` | boolean | `false` | `true` |
| `fail_fall` | boolean | `false` | `false` |
| `fail_stance_loss` | boolean | `false` | `true` |
| `fail_reason` | null or string | `null` | `stance_loss` |

An episode in this file has no other keys. There is no `scenario` field on the episode: the scenario is stored above, on the sweep object (`scenario`: `nominal` or `moderate`).

Across every episode of both scenarios, `fail_reason` takes only three values: `null`, `stance_loss`, `fall`.

## `reports/outcome_thresholds.json`

The root is an object. The only nested object is `failure_criterion`.

| Field | Type | Value |
| --- | --- | --- |
| `n_nominal` | integer | `200` |
| `n_standing` | integer | `200` |
| `z_ref` | number | `0.2440828391927459` |
| `z_settle_mean` | number | `0.24363440479537132` |
| `z_settle_min` | number | `0.24132307227128694` |
| `z_settle_max` | number | `0.24630712325938614` |
| `z_settle_std` | number | `0.0009723104052120386` |
| `z_threshold_rule` | string | `min(nominal) - 3σ` |
| `z_threshold` | number | `0.23840614105565083` |
| `slip_mean` | number | `0.0372862175631595` |
| `slip_min` | number | `0.026096063048556432` |
| `slip_max` | number | `0.048486919219676586` |
| `slip_std` | number | `0.0045421121268086765` |
| `slip_threshold_rule` | string | `max(nominal) + 3σ` |
| `slip_threshold` | number | `0.06211325560010261` |
| `failure_criterion` | object | see the table below |
| `post_impact_measure_s` | number | `0.4` |
| `seed` | integer | `42` |
| `created_at` | string | `2026-11-22T18:37:00.604341+00:00` |

### `failure_criterion`

| Field | Type | Value |
| --- | --- | --- |
| `fail` | string | `fall OR stance_loss` |
| `fall` | string | `abs(roll)>45° or abs(pitch)>45° or body z<0.12 m` |
| `stance_loss` | string | `z_settle < z_threshold OR slip > slip_threshold` |
| `slip` | string | `max foot xy displacement relative to body` |
| `measure` | string | `0.40s after impulse_end` |
