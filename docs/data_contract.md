# Контракт данных

Источник: `reports/batch_failures.json` и `reports/outcome_thresholds.json` в том виде, в каком они лежат в репозитории. Поля, которых в файлах нет, сюда не внесены.

Эпизод разобран по записям `outcomes`. У всех 10200 исходов в `parameter_sweeps` и всех 10200 исходов в `parameter_sweeps_by_scenario.moderate` один и тот же набор ключей. Первые два эпизода уровня `com_x` / `zero` сценария `nominal` не содержат сбоя; пример со сбоем взят из `floor_contact` / `soft` того же сценария.

## `reports/batch_failures.json`

Корень — объект из 39 полей.

| Поле | Тип | Что лежит |
| --- | --- | --- |
| `n_episodes` | integer | `80` |
| `n_envs` | integer | `80` |
| `n_failures` | integer | `0` |
| `n_survived` | integer | `80` |
| `failure_rate` | number | `0.0` |
| `mode` | string | `baseline` |
| `seed` | integer | `42` |
| `elapsed_s` | number | секунды прогона |
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
| `target_qpos` | array of number | 12 чисел |
| `failed` | object | сводка по упавшим, см. ниже |
| `survived` | object | сводка по устоявшим, см. ниже |
| `failures` | array | пустой массив, элементов нет |
| `parameter_sweeps` | object | OFAT-сценарий `nominal` |
| `run_mode` | string | `ofat` |
| `pd` | string | `baseline` |
| `scenario` | string | `default` |
| `scenario_label` | string | `базовый: трение и боковой удар (Fy до ±350 Н)` |
| `friction_range` | array of number | `[0.05, 1.0]` |
| `lateral_force_range` | array of number | `[-350.0, 350.0]` |
| `stance_hold` | object | `friction`, `lateral_force`, `n_episodes`, `n_failures`, `failure_rate` |
| `baseline_grid_path` | string | `reports\baseline_grid.csv` |
| `baseline_grid` | array of object | 42 строки сетки, не эпизоды OFAT |
| `created_at` | string | `2026-11-22T18:09:35.278393+00:00` |
| `config` | object | снимок конфигурации, см. ниже |
| `config_hash` | string | `f920c32514104f9b` |
| `parameter_sweeps_by_scenario` | object | ключи `nominal` и `moderate` |
| `parameter_sweeps_moderate` | object | тот же сценарий, что `parameter_sweeps_by_scenario.moderate` |

Отдельного массива эпизодов в корне нет. Поэпизодные записи лежат здесь:

`parameter_sweeps.parameters[].levels[].outcomes[]`

Та же форма у `parameter_sweeps_by_scenario.nominal`, `parameter_sweeps_by_scenario.moderate` и `parameter_sweeps_moderate`.

### Сводки `failed` и `survived`

У обоих объектов два ключа: `friction` и `lateral_force`. У каждой из этих сводок четыре поля.

| Поле | Тип |
| --- | --- |
| `mean` | number или null |
| `min` | number или null |
| `max` | number или null |
| `count` | integer |

В текущем файле у `failed` все `mean` / `min` / `max` равны null, `count` равен 0. У `survived` числа заполнены, `count` равен 80.

### `config`

| Поле | Тип | Значение в файле |
| --- | --- | --- |
| `model_path` | string | путь к `unitree_go2\scene.xml` |
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

### Уровень фактора, внутри которого лежит эпизод

Путь: `parameter_sweeps.parameters[]` — объект с полями `id`, `name`, `baseline`, `levels`.

Один элемент `levels[]`:

| Поле | Тип | Примечание по файлу |
| --- | --- | --- |
| `id` | string | например `zero` |
| `label` | string | например `базовый 0 м` |
| `factor` | object | набор ключей зависит от фактора |
| `n_episodes` | integer | `300` |
| `n_failures` | integer | |
| `n_survived` | integer | |
| `failure_rate` | number | |
| `failure_rate_ci95` | array of number, длина 2 | |
| `n_sag` | integer | |
| `sag_rate` | number | |
| `sag_rate_ci95` | array of number, длина 2 | |
| `fail_reasons` | object | счётчики причин; у нулевого уровня бывает `{}` |
| `n_tilt` | integer | |
| `tilt_rate` | number | |
| `tilt_rate_ci95` | array of number, длина 2 | |
| `n_stance_loss` | integer | |
| `stance_loss_rate` | number | |
| `stance_loss_rate_ci95` | array of number, длина 2 | |
| `n_before_impact` | integer | |
| `n_at_or_after_impact` | integer | |
| `min_z_mean` | number | |
| `max_abs_roll_mean` | number | |
| `elapsed_s` | number | |
| `n_workers` | integer | |
| `n_chunks` | integer | |
| `outcomes` | array of object | эпизоды, см. следующий раздел |
| `delta_vs_baseline` | number | |
| `mcnemar` | object или null | у базового уровня `com_x` / `zero` стоит null |
| `p_mcnemar` | number или null | |
| `p_holm` | number или null | |
| `significant` | boolean | |
| `effect_label` | string | в файле встречаются `норма`, `в пределах шума`, `значимо` |

`fail_reasons` — не поле эпизода. Это объект уровня. В файле встречаются пустой объект, `{"stance_loss": 300}` и `{"fall": 72, "stance_loss": 61}`.

Если `mcnemar` — объект, у него такие ключи: `test` (string, значение `mcnemar_exact`), `n_paired`, `n00`, `n01`, `n10`, `n11`, `n_discordant` (integer), `pvalue` (number).

### Один эпизод (`outcomes[]`)

| Поле | Тип | Пример без сбоя (`nominal`, `com_x` / `zero`, эпизоды 1 и 2) | Пример со сбоем (`nominal`, `floor_contact` / `soft`, эпизод 1) |
| --- | --- | --- | --- |
| `episode` | integer | `1`, затем `2` | `1` |
| `seed` | integer | `42000127`, затем `42000128` | `42000127` |
| `failed` | boolean | `false` | `true` |
| `fail_fall` | boolean | `false` | `false` |
| `fail_stance_loss` | boolean | `false` | `true` |
| `fail_reason` | null или string | `null` | `stance_loss` |

Других ключей у эпизода в этом файле нет. Поля `scenario` на эпизоде нет: сценарий записан выше, на объекте свипа (`scenario`: `nominal` или `moderate`).

По всем эпизодам обоих сценариев `fail_reason` принимает только три значения: `null`, `stance_loss`, `fall`.

## `reports/outcome_thresholds.json`

Корень — объект. Вложенный объект один: `failure_criterion`.

| Поле | Тип | Значение |
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
| `failure_criterion` | object | см. таблицу ниже |
| `post_impact_measure_s` | number | `0.4` |
| `seed` | integer | `42` |
| `created_at` | string | `2026-11-22T18:37:00.604341+00:00` |

### `failure_criterion`

| Поле | Тип | Значение |
| --- | --- | --- |
| `fail` | string | `fall OR stance_loss` |
| `fall` | string | `abs(roll)>45° or abs(pitch)>45° or body z<0.12 m` |
| `stance_loss` | string | `z_settle < z_threshold OR slip > slip_threshold` |
| `slip` | string | `max foot xy displacement relative to body` |
| `measure` | string | `0.40s after impulse_end` |
