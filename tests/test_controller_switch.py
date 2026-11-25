"""Switching the controller changes the command and the outcome on the same seed."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco

import controllers
import step3_batch_stress as s3
from controllers import STANCE_SHIFT_RAD, make_controller, prepare_controller_spec


class ControllerSwitchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(s3.MODEL_PATH)
        cls.data = mujoco.MjData(cls.model)
        floor_id = s3.find_floor_geom_id(cls.model)
        foot_ids = s3.find_foot_geom_ids(cls.model)
        cls.torso_id = s3.find_torso_body_id(cls.model)
        cls.friction_geoms = [floor_id] + [g for g in foot_ids if g != floor_id]
        s3.reset_to_stance(cls.model, cls.data)
        cls.target = np.array(cls.data.qpos[7:], copy=True)
        cls.kp = float(s3.PD_MODES["baseline"]["kp"])
        cls.kd = float(s3.PD_MODES["baseline"]["kd"])
        s3.load_outcome_thresholds()

    def test_default_pd_matches_and_alternative_changes_same_seed(self) -> None:
        pd = make_controller("pd", self.kp, self.kd)
        alt = make_controller("stance_shift", self.kp, self.kd)
        weak = make_controller("pd", s3.PD_MODES["weak_pd"]["kp"], s3.PD_MODES["weak_pd"]["kd"])
        self.assertEqual(pd.name, "pd")
        self.assertEqual(alt.name, "stance_shift")
        self.assertEqual(weak.kp, 25.0)
        self.assertEqual(weak.kd, 1.5)

        obs = {"q": self.target.copy(), "v": np.zeros_like(self.target)}
        cmd_pd = np.asarray(pd.act(obs, 1, float(self.model.opt.timestep), self.target))
        cmd_alt = np.asarray(alt.act(obs, 1, float(self.model.opt.timestep), self.target))
        cmd_gap = float(np.max(np.abs(cmd_pd - cmd_alt)))
        self.assertGreater(cmd_gap, 1.0)
        self.assertAlmostEqual(cmd_gap, self.kp * STANCE_SHIFT_RAD, places=6)

        s3.reset_to_stance(self.model, self.data)
        self.data.qvel[6:] = 0.05
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        s3.apply_pd(
            self.model,
            self.data,
            self.target,
            kp=self.kp,
            kd=self.kd,
            rng=rng_a,
            obs_noise_q=0.01,
            obs_noise_v=0.02,
            obs_bias_q=0.03,
        )
        via_apply = np.array(self.data.ctrl, copy=True)
        via_iface = s3.command_from_controller(
            self.model,
            self.data,
            make_controller("pd", self.kp, self.kd),
            self.target,
            0,
            float(self.model.opt.timestep),
            rng_b,
            0.01,
            0.02,
            0.03,
        )
        np.testing.assert_allclose(via_apply, via_iface, atol=1e-12)

        noisy_pd = s3.command_from_controller(
            self.model,
            self.data,
            pd,
            self.target,
            1,
            float(self.model.opt.timestep),
            np.random.default_rng(11),
            0.01,
            0.02,
            0.0,
        )
        noisy_alt = s3.command_from_controller(
            self.model,
            self.data,
            alt,
            self.target,
            1,
            float(self.model.opt.timestep),
            np.random.default_rng(11),
            0.01,
            0.02,
            0.0,
        )
        np.testing.assert_allclose(
            noisy_alt - noisy_pd,
            np.full(self.target.shape, self.kp * STANCE_SHIFT_RAD),
            atol=1e-12,
        )
        s3.reset_to_stance(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        s3.apply_pd(
            self.model,
            self.data,
            self.target,
            kp=self.kp,
            kd=self.kd,
            gravity_ff=True,
            ff_ctrl=np.full(self.target.shape, 0.1),
        )
        ff_apply = np.array(self.data.ctrl, copy=True)
        ff_iface = s3.command_from_controller(
            self.model,
            self.data,
            make_controller("pd", self.kp, self.kd),
            self.target,
            1,
            float(self.model.opt.timestep),
            None,
            0.0,
            0.0,
            0.0,
            gravity_ff=True,
            ff_ctrl=np.full(self.target.shape, 0.1),
        )
        np.testing.assert_allclose(ff_apply, ff_iface, atol=1e-12)

        common = dict(
            episode=4,
            jitter=True,
            keep_window=False,
            record_trace=False,
        )
        base = s3.simulate_episode(
            self.model,
            self.data,
            self.target,
            self.friction_geoms,
            self.torso_id,
            0.8,
            120.0,
            self.kp,
            self.kd,
            controller=make_controller("pd", self.kp, self.kd),
            **common,
        )
        shifted = s3.simulate_episode(
            self.model,
            self.data,
            self.target,
            self.friction_geoms,
            self.torso_id,
            0.8,
            120.0,
            self.kp,
            self.kd,
            controller=make_controller("stance_shift", self.kp, self.kd),
            **common,
        )
        delayed = s3.simulate_episode(
            self.model,
            self.data,
            self.target,
            self.friction_geoms,
            self.torso_id,
            0.8,
            120.0,
            self.kp,
            self.kd,
            controller=make_controller("stance_shift", self.kp, self.kd),
            factor={"kind": "delay", "steps": 15},
            **common,
        )
        held = s3.simulate_episode(
            self.model,
            self.data,
            self.target,
            self.friction_geoms,
            self.torso_id,
            0.8,
            120.0,
            self.kp,
            self.kd,
            controller=make_controller("pd", self.kp, self.kd),
            factor={"kind": "ctrl_hold", "steps": 20},
            **common,
        )
        biased = s3.simulate_episode(
            self.model,
            self.data,
            self.target,
            self.friction_geoms,
            self.torso_id,
            0.8,
            120.0,
            self.kp,
            self.kd,
            controller=make_controller("pd", self.kp, self.kd),
            factor={"kind": "bias", "q": 0.05, "drift": 0.0},
            **common,
        )
        noisy = s3.simulate_episode(
            self.model,
            self.data,
            self.target,
            self.friction_geoms,
            self.torso_id,
            0.8,
            120.0,
            self.kp,
            self.kd,
            controller=make_controller("stance_shift", self.kp, self.kd),
            factor={"kind": "noise", "q": 0.02, "v": 0.05},
            **common,
        )
        try:
            s3.apply_torque_limit(self.model, 0.05)
            clipped_pd = s3.simulate_episode(
                self.model,
                self.data,
                self.target,
                self.friction_geoms,
                self.torso_id,
                0.8,
                120.0,
                self.kp,
                self.kd,
                controller=make_controller("pd", self.kp, self.kd),
                **common,
            )
            clipped_alt = s3.simulate_episode(
                self.model,
                self.data,
                self.target,
                self.friction_geoms,
                self.torso_id,
                0.8,
                120.0,
                self.kp,
                self.kd,
                controller=make_controller("stance_shift", self.kp, self.kd),
                **common,
            )
        finally:
            s3.apply_torque_limit(self.model, 1.0)

        self.assertEqual(base["seed"], shifted["seed"])
        z_gap = abs(base["z_end"] - shifted["z_end"])
        self.assertGreater(z_gap, 1e-3)
        self.assertNotEqual(base["failed"], shifted["failed"])
        self.assertTrue(
            abs(delayed["z_end"] - shifted["z_end"]) > 1e-4
            or delayed["n_steps"] != shifted["n_steps"]
            or delayed["failed"] != shifted["failed"]
        )
        self.assertTrue(
            abs(held["z_end"] - base["z_end"]) > 1e-4
            or held["n_steps"] != base["n_steps"]
            or held["failed"] != base["failed"]
        )
        self.assertTrue(
            abs(biased["z_end"] - base["z_end"]) > 1e-4
            or biased["failed"] != base["failed"]
        )
        self.assertGreater(abs(noisy["max_abs_ctrl_raw"] - shifted["max_abs_ctrl_raw"]), 0.5)
        self.assertGreater(clipped_pd["saturation_frac"], 0.5)
        self.assertGreater(clipped_alt["saturation_frac"], 0.5)
        self.assertTrue(clipped_pd["hit_limit"])
        self.assertTrue(clipped_alt["hit_limit"])
        print(
            f"cmd_gap={cmd_gap:.4f} seed={base['seed']} "
            f"z {base['z_end']:.4f}->{shifted['z_end']:.4f} "
            f"fail {int(base['failed'])}->{int(shifted['failed'])} "
            f"raw {base['max_abs_ctrl_raw']:.4f}->{shifted['max_abs_ctrl_raw']:.4f} "
            f"delay_z={delayed['z_end']:.4f} hold_z={held['z_end']:.4f} "
            f"bias_z={biased['z_end']:.4f} noise_raw={noisy['max_abs_ctrl_raw']:.4f} "
            f"clip_sat {clipped_pd['saturation_frac']:.3f}/{clipped_alt['saturation_frac']:.3f}"
        )


    def test_external_file_and_module_controller(self) -> None:
        source = textwrap.dedent(
            """
            import numpy as np

            class ExternalShift:
                def __init__(self, kp, kd):
                    self.kp = float(kp)
                    self.kd = float(kd)
                    self.name = "external_shift"

                def reset(self):
                    return None

                def act(self, obs, step, dt, target_qpos):
                    q = np.asarray(obs["q"], dtype=float)
                    v = np.asarray(obs["v"], dtype=float)
                    target = np.asarray(target_qpos, dtype=float) + 0.2
                    return self.kp * (target - q) - self.kd * v

            def compute_fn(obs, step, dt, target_qpos):
                q = np.asarray(obs["q"], dtype=float)
                v = np.asarray(obs["v"], dtype=float)
                target = np.asarray(target_qpos, dtype=float) + 0.2
                return 40.0 * (target - q) - 3.0 * v

            def bad_cmd(obs, step, dt, target_qpos):
                return np.zeros(3)
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "external_ctrl.py"
            path.write_text(source, encoding="utf-8")
            mod_path = folder / "extmod_demo.py"
            mod_path.write_text(source, encoding="utf-8")
            spec = f"{path}:ExternalShift"
            canonical = controllers.canonical_controller_spec(spec)
            self.assertTrue(canonical.endswith("external_ctrl.py:ExternalShift"))
            old_argv = sys.argv
            try:
                sys.argv = ["step3_batch_stress.py", "--controller", spec]
                parsed = s3.parse_args()
            finally:
                sys.argv = old_argv
            self.assertEqual(parsed.controller, spec)

            external = make_controller(spec, self.kp, self.kd)
            fn_ctrl = make_controller(f"{path}:compute_fn", self.kp, self.kd)
            obs = {"q": self.target.copy(), "v": np.zeros_like(self.target)}
            dt = float(self.model.opt.timestep)
            cmd_pd = np.asarray(
                make_controller("pd", self.kp, self.kd).act(obs, 1, dt, self.target)
            )
            cmd_ext = np.asarray(external.act(obs, 1, dt, self.target))
            cmd_fn = np.asarray(fn_ctrl.act(obs, 1, dt, self.target))
            cmd_gap = float(np.max(np.abs(cmd_ext - cmd_pd)))
            self.assertAlmostEqual(cmd_gap, self.kp * 0.2, places=6)
            np.testing.assert_allclose(cmd_fn, cmd_ext, atol=1e-12)

            sys.path.insert(0, str(folder))
            try:
                mod_ctrl = make_controller("extmod_demo:compute_fn", self.kp, self.kd)
            finally:
                sys.path.pop(0)
            np.testing.assert_allclose(
                np.asarray(mod_ctrl.act(obs, 1, dt, self.target)),
                cmd_ext,
                atol=1e-12,
            )

            with self.assertRaises(ValueError) as missing_file:
                make_controller(str(folder / "no_such_controller.py") + ":ExternalShift", 1, 1)
            self.assertIn("not found", str(missing_file.exception))
            with self.assertRaises(ValueError) as missing_attr:
                make_controller(f"{path}:NoSuchCtrl", self.kp, self.kd)
            self.assertIn("no attribute", str(missing_attr.exception))
            with self.assertRaises(ValueError) as broken:
                make_controller("not-a-controller", self.kp, self.kd)
            self.assertIn("invalid controller spec", str(broken.exception))

            bad = make_controller(f"{path}:bad_cmd", self.kp, self.kd)
            s3.reset_to_stance(self.model, self.data)
            with self.assertRaises(ValueError) as bad_vec:
                s3.command_from_controller(
                    self.model,
                    self.data,
                    bad,
                    self.target,
                    1,
                    dt,
                    None,
                    0.0,
                    0.0,
                    0.0,
                )
            self.assertIn("expected", str(bad_vec.exception))
            self.assertIn(str(int(self.model.nu)), str(bad_vec.exception))

            common = dict(episode=4, jitter=True, keep_window=False, record_trace=False)
            base = s3.simulate_episode(
                self.model,
                self.data,
                self.target,
                self.friction_geoms,
                self.torso_id,
                0.8,
                120.0,
                self.kp,
                self.kd,
                controller_name="pd",
                **common,
            )
            shifted = s3.simulate_episode(
                self.model,
                self.data,
                self.target,
                self.friction_geoms,
                self.torso_id,
                0.8,
                120.0,
                self.kp,
                self.kd,
                controller_name=canonical,
                **common,
            )
            delayed = s3.simulate_episode(
                self.model,
                self.data,
                self.target,
                self.friction_geoms,
                self.torso_id,
                0.8,
                120.0,
                self.kp,
                self.kd,
                controller_name=canonical,
                factor={"kind": "delay", "steps": 15},
                **common,
            )
            held = s3.simulate_episode(
                self.model,
                self.data,
                self.target,
                self.friction_geoms,
                self.torso_id,
                0.8,
                120.0,
                self.kp,
                self.kd,
                controller_name=canonical,
                factor={"kind": "ctrl_hold", "steps": 20},
                **common,
            )
            biased = s3.simulate_episode(
                self.model,
                self.data,
                self.target,
                self.friction_geoms,
                self.torso_id,
                0.8,
                120.0,
                self.kp,
                self.kd,
                controller_name=canonical,
                factor={"kind": "bias", "q": 0.05, "drift": 0.0},
                **common,
            )
            noisy = s3.simulate_episode(
                self.model,
                self.data,
                self.target,
                self.friction_geoms,
                self.torso_id,
                0.8,
                120.0,
                self.kp,
                self.kd,
                controller_name=canonical,
                factor={"kind": "noise", "q": 0.02, "v": 0.05},
                **common,
            )
            try:
                s3.apply_torque_limit(self.model, 0.05)
                clipped = s3.simulate_episode(
                    self.model,
                    self.data,
                    self.target,
                    self.friction_geoms,
                    self.torso_id,
                    0.8,
                    120.0,
                    self.kp,
                    self.kd,
                    controller_name=canonical,
                    **common,
                )
            finally:
                s3.apply_torque_limit(self.model, 1.0)

            payload = {
                "specs": [(4, 0.8, 120.0)],
                "kp": self.kp,
                "kd": self.kd,
                "keep_windows": False,
                "target_qpos": self.target.tolist(),
                "factor": None,
                "stance_ref_z": s3.STANCE_REF.get("z"),
                "stance_loss_frac": s3.STANCE_REF.get("frac"),
                "stance_thresh": {
                    key: value
                    for key, value in dict(s3.STANCE_THRESH).items()
                    if key != "derivation"
                },
                "impulse_cfg": dict(s3.IMPULSE_CFG),
                "controller": canonical,
            }
            with ProcessPoolExecutor(max_workers=1) as pool:
                remote = pool.submit(s3._run_chunk, payload).result(timeout=180)
            remote_ep = remote["episodes"][0]

            self.assertEqual(base["seed"], shifted["seed"])
            self.assertEqual(int(remote_ep["seed"]), int(shifted["seed"]))
            self.assertAlmostEqual(float(remote_ep["z_end"]), float(shifted["z_end"]), places=6)
            self.assertEqual(bool(remote_ep["failed"]), bool(shifted["failed"]))
            self.assertNotEqual(base["failed"], shifted["failed"])
            self.assertGreater(abs(base["z_end"] - shifted["z_end"]), 1e-3)
            self.assertGreater(abs(delayed["z_end"] - shifted["z_end"]), 1e-3)
            self.assertGreater(abs(held["z_end"] - shifted["z_end"]), 1e-3)
            self.assertGreater(abs(biased["z_end"] - shifted["z_end"]), 1e-3)
            self.assertGreater(abs(noisy["max_abs_ctrl_raw"] - shifted["max_abs_ctrl_raw"]), 0.5)
            self.assertGreater(clipped["saturation_frac"], 0.5)
            self.assertTrue(clipped["hit_limit"])
            print(
                f"ext_cmd_gap={cmd_gap:.4f} seed={shifted['seed']} "
                f"z {base['z_end']:.4f}->{shifted['z_end']:.4f} "
                f"fail {int(base['failed'])}->{int(shifted['failed'])} "
                f"worker_z={float(remote_ep['z_end']):.4f} "
                f"delay_z={delayed['z_end']:.4f} hold_z={held['z_end']:.4f} "
                f"bias_z={biased['z_end']:.4f} noise_raw={noisy['max_abs_ctrl_raw']:.4f} "
                f"clip_sat={clipped['saturation_frac']:.3f}"
            )


    def test_parallel_batch_file_and_module_outside_path(self) -> None:
        source = textwrap.dedent(
            """
            import numpy as np

            class ExternalShift:
                def __init__(self, kp, kd):
                    self.kp = float(kp)
                    self.kd = float(kd)
                    self.name = "external_shift"

                def act(self, obs, step, dt, target_qpos):
                    q = np.asarray(obs["q"], dtype=float)
                    v = np.asarray(obs["v"], dtype=float)
                    target = np.asarray(target_qpos, dtype=float) + 0.2
                    return self.kp * (target - q) - self.kd * v
            """
        )
        reports = Path(__file__).resolve().parents[1] / "reports"
        before = _reports_snapshot(reports)
        old_cwd = os.getcwd()
        old_controller = s3.ACTIVE_CONTROLLER
        old_root = controllers.CONTROLLER_IMPORT_ROOT
        old_chunk = s3.CHUNK_SIZE
        specs = [(4, 0.8, 120.0), (5, 0.8, 80.0)]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                file_path = folder / "external_ctrl.py"
                file_path.write_text(source, encoding="utf-8")
                module_path = folder / "ext_batch_mod.py"
                module_path.write_text(source, encoding="utf-8")
                self.assertNotIn(str(folder), sys.path)

                os.chdir(folder)
                file_spec = controllers.canonical_controller_spec("external_ctrl.py:ExternalShift")
                os.chdir(old_cwd)
                file_location = file_spec.rsplit(":", 1)[0]
                self.assertTrue(Path(file_location).is_absolute())
                self.assertTrue(Path(file_location).is_file())

                s3.CHUNK_SIZE = 1
                s3.ACTIVE_CONTROLLER = file_spec
                controllers.CONTROLLER_IMPORT_ROOT = None
                os.chdir(Path(tempfile.gettempdir()))
                file_batch = s3.run_stress_batch_parallel(
                    specs, self.kp, self.kd, False, self.target
                )
                os.chdir(old_cwd)

                self.assertTrue(file_batch["used_process_pool"])
                self.assertEqual(file_batch["n_chunks"], 2)
                self.assertEqual(file_batch["n_episodes"], 2)
                by_ep = {int(ep["episode"]): ep for ep in file_batch["episodes"]}
                local = {}
                for episode, mu, fy in specs:
                    local[episode] = s3.simulate_episode(
                        self.model,
                        self.data,
                        self.target,
                        self.friction_geoms,
                        self.torso_id,
                        mu,
                        fy,
                        self.kp,
                        self.kd,
                        episode=episode,
                        jitter=True,
                        controller_name=file_spec,
                    )
                    self.assertAlmostEqual(
                        float(by_ep[episode]["z_end"]),
                        float(local[episode]["z_end"]),
                        places=6,
                    )
                    self.assertEqual(int(by_ep[episode]["seed"]), int(local[episode]["seed"]))

                if str(folder) in sys.path:
                    sys.path.remove(str(folder))
                sys.path.insert(0, str(folder))
                mod_spec, mod_root = prepare_controller_spec("ext_batch_mod:ExternalShift")
                while str(folder) in sys.path:
                    sys.path.remove(str(folder))
                while str(Path(mod_root).resolve()) in sys.path:
                    sys.path.remove(str(Path(mod_root).resolve()))
                self.assertNotIn(str(Path(mod_root).resolve()), sys.path)
                s3.ACTIVE_CONTROLLER = mod_spec
                controllers.CONTROLLER_IMPORT_ROOT = mod_root
                os.chdir(Path(tempfile.gettempdir()))
                mod_batch = s3.run_stress_batch_parallel(
                    specs, self.kp, self.kd, False, self.target
                )
                os.chdir(old_cwd)
                mod_by_ep = {int(ep["episode"]): ep for ep in mod_batch["episodes"]}
                self.assertTrue(mod_batch["used_process_pool"])
                self.assertEqual(mod_batch["n_chunks"], 2)
                for episode in by_ep:
                    self.assertAlmostEqual(
                        float(mod_by_ep[episode]["z_end"]),
                        float(by_ep[episode]["z_end"]),
                        places=6,
                    )
                    self.assertEqual(
                        bool(mod_by_ep[episode]["failed"]),
                        bool(by_ep[episode]["failed"]),
                    )

                with self.assertRaises(ValueError) as missing_mod:
                    make_controller("no_such_ext_mod_zz:ExternalShift", self.kp, self.kd)
                self.assertIn("not found", str(missing_mod.exception))
                sys.path.insert(0, str(folder))
                try:
                    with self.assertRaises(ValueError) as missing_attr:
                        prepare_controller_spec("ext_batch_mod:NoSuchCtrl")
                    self.assertIn("no attribute", str(missing_attr.exception))
                finally:
                    while str(folder) in sys.path:
                        sys.path.remove(str(folder))

                print(
                    f"pool_file workers={file_batch['n_workers']} chunks={file_batch['n_chunks']} "
                    f"ep4 z={by_ep[4]['z_end']:.4f} fail={int(by_ep[4]['failed'])} "
                    f"ep5 z={by_ep[5]['z_end']:.4f} seed={by_ep[4]['seed']} "
                    f"pool_mod z4={mod_by_ep[4]['z_end']:.4f} "
                    f"root={mod_root}"
                )
        finally:
            os.chdir(old_cwd)
            s3.ACTIVE_CONTROLLER = old_controller
            controllers.CONTROLLER_IMPORT_ROOT = old_root
            s3.CHUNK_SIZE = old_chunk
        self.assertEqual(_reports_snapshot(reports), before)


def _reports_snapshot(reports: Path) -> list[tuple[str, int]]:
    if not reports.exists():
        return []
    rows = []
    for path in sorted(reports.iterdir()):
        if path.is_file():
            rows.append((path.name, path.stat().st_size))
    return rows


if __name__ == "__main__":
    unittest.main()
