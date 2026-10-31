"""Regression checks for disk counts, grasp handedness, and dataset labels."""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import ocbench
from ocbench.oracles.controllers.hanoi import HanoiAutoresetController, HanoiController
from scripts import generate_hanoi_lerobot as generator
from scripts.generate_hanoi_lerobot import (
    AUTORESET_CPU_ENV_NAME,
    SUBTASK_CPU_ENV_NAME,
    _subtask_descriptions,
    parse_args,
)


class HanoiGenerationTest(unittest.TestCase):
    def test_disk_counts_and_labels(self):
        for count in (1, 2, 3):
            for name in ('hanoi-cpu-triple-task2-v0', SUBTASK_CPU_ENV_NAME, AUTORESET_CPU_ENV_NAME):
                with self.subTest(count=count, name=name):
                    env = ocbench.make(name, num_disks=count)
                    try:
                        _, info = env.reset(seed=4)
                        e = env.unwrapped
                        self.assertEqual(len(e._disk_geom_ids_list), count)
                        self.assertEqual(len(e._disk_colors), count)
                        self.assertNotIn(f'privileged/disk_{count}_pos', info)
                        if e.cur_task_info['init_mode'] == 'stack':
                            self.assertTrue(e._is_tower_on_peg(1))
                        labels = _subtask_descriptions(name, count)
                        self.assertEqual(len(labels), count + (count * count if 'autoreset' in name else 0))
                    finally:
                        env.close()

    def test_invalid_disk_counts(self):
        for count in (0, 4, -1, 1.5, True):
            with self.assertRaises(ValueError):
                ocbench.make(SUBTASK_CPU_ENV_NAME, num_disks=count)
        self.assertEqual(parse_args(['--num-disks', '2']).num_disks, 2)
        self.assertIn('2-disk', parse_args(['--num-disks', '2']).task_description)

    def test_single_disk_autoreset(self):
        env = ocbench.make(AUTORESET_CPU_ENV_NAME, num_disks=1)
        try:
            ob, info = env.reset(seed=0)
            controller = HanoiAutoresetController(env)
            controller.reset(ob, info)
            self.assertEqual(controller._reset_slots, {0: 0})
            controller._phase = 'reset'
            self.assertEqual(controller.subtask_index, 1)
        finally:
            env.close()

    def test_both_grasp_sides_keep_hole_toward_camera(self):
        env = ocbench.make(SUBTASK_CPU_ENV_NAME, num_disks=2)
        try:
            ob, info = env.reset(seed=0)
            controller = HanoiController(env)
            controller.reset(ob, info)
            primitive = controller._agent
            target = controller._target_info(controller._active_subgoal, info)
            for sign in (-1, 1):
                with (
                    patch('numpy.random.choice', return_value=sign),
                    patch.object(primitive, 'compute_keyframes', wraps=primitive.compute_keyframes) as keyframes,
                ):
                    primitive.reset(ob, info, target)
                    plan = keyframes.call_args.args[0]
                for key, center in (
                    ('disk_initial', info[f'privileged/disk_{target["disk"]}_pos']),
                    ('disk_goal', target['disk_goal_pos']),
                ):
                    pose = plan[key]
                    hole_direction = center[:2] - pose.translation()[:2]
                    yaw = primitive.get_yaw(pose)
                    camera_direction = -np.array([np.cos(yaw), np.sin(yaw)])
                    cosine = hole_direction @ camera_direction / np.linalg.norm(hole_direction)
                    self.assertGreater(cosine, 0.999)
        finally:
            env.close()

    def test_generator_rejects_obscured_insertions(self):
        with tempfile.TemporaryDirectory() as directory:
            config = generator.WorkerConfig(
                temp_dir=directory,
                env_name=SUBTASK_CPU_ENV_NAME,
                num_disks=1,
                fps=30,
                physics_substeps=20,
                max_steps=2000,
                cameras=(),
                width=96,
                height=96,
                successful_only=False,
                max_attempts=1,
                subtask=False,
                with_reward=False,
                with_done=False,
            )
            generator._init_worker(config)
            try:
                with (
                    patch.object(generator._WORKER_ENV.unwrapped, 'is_disk_hole_visible', return_value=False),
                    self.assertRaisesRegex(RuntimeError, 'wrist_visible=False'),
                ):
                    generator._rollout_episode(0, 1)
            finally:
                generator._WORKER_ENV.close()

    def test_visibility_rejects_camera_facing_away(self):
        env = ocbench.make(SUBTASK_CPU_ENV_NAME, num_disks=1)
        try:
            env.reset(seed=0)
            e = env.unwrapped
            camera = e._data.camera('ur5e/wrist')
            center = e._data.joint('disk_joint_0').qpos[:3]
            camera.xpos[:] = center + [0, 0, 0.5]
            camera.xmat[:] = np.eye(3).ravel()
            self.assertTrue(e.is_disk_hole_visible(0))
            camera.xmat[:] = np.diag([1, -1, -1]).ravel()
            self.assertFalse(e.is_disk_hole_visible(0))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
