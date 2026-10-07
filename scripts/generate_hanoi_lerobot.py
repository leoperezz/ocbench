#!/usr/bin/env python3
"""Generate OCBench Hanoi RGB demonstrations as H.264 LeRobot v3 videos.

Simulation workers only generate episodes.  The parent process is the sole
LeRobotDataset writer, avoiding concurrent writes to Parquet, videos, and
metadata.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

ENV_NAME = 'hanoi-cpu-triple-task2-v0'
TASK_DESCRIPTION = 'Solve the randomized three-disk Tower of Hanoi.'
DEFAULT_FPS = 30
DEFAULT_MAX_STEPS = 7000
DEFAULT_CAMERAS = ('front', 'side', 'wrist')
CAMERA_RENDER_NAMES = {
    'wrist': 'ur5e/wrist',
}


@dataclass(frozen=True)
class WorkerConfig:
    temp_dir: str
    fps: int
    physics_substeps: int
    max_steps: int
    cameras: tuple[str, ...]
    width: int
    height: int
    successful_only: bool
    max_attempts: int


_WORKER_CONFIG: WorkerConfig | None = None
_WORKER_ENV = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate hanoi-cpu-triple-task2-v0 demonstrations as a LeRobot v3 dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--output-dir', type=Path, default=Path('data/lerobot/hanoi-triple-task2-rgb'))
    parser.add_argument(
        '--repo-id', default='local/hanoi-triple-task2-rgb', help='LeRobot/Hugging Face dataset ID.'
    )
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--workers', type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS)
    parser.add_argument('--physics-substeps', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        '--cameras',
        nargs='+',
        choices=('front', 'front_pixels', 'side', 'wrist'),
        default=DEFAULT_CAMERAS,
        help='RGB camera streams to encode as MP4 videos.',
    )
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument(
        '--video',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Store cameras as MP4 video. Image-only output is not supported by this generator.',
    )
    parser.add_argument(
        '--video-encoder',
        choices=('h264', 'h264_nvenc'),
        default='h264',
        help='H.264 encoder. Use h264_nvenc only when NVIDIA NVENC is available.',
    )
    parser.add_argument('--image-writer-threads', type=int, default=4)
    parser.add_argument('--successful-only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-attempts', type=int, default=10, help='Attempts per requested successful episode.')
    parser.add_argument('--overwrite', action='store_true', help='Replace output-dir if it already exists.')
    parser.add_argument('--push-to-hub', action='store_true')
    parser.add_argument('--private', action='store_true', help='Make the Hub dataset private when pushing.')
    args = parser.parse_args(argv)

    for name in ('num_episodes', 'workers', 'fps', 'physics_substeps', 'max_steps', 'width', 'height', 'max_attempts'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be greater than zero')
    if len(set(args.cameras)) != len(args.cameras):
        parser.error('--cameras cannot contain duplicates')
    if not args.video:
        parser.error('--no-video is not supported; this generator always creates LeRobot MP4 videos')
    if args.private and not args.push_to_hub:
        parser.error('--private requires --push-to-hub')
    return args


def _require_lerobot():
    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            'LeRobot with dataset support is required. Install the data extra first:\n'
            '  pip install -e ".[data]"\n'
            'or:\n'
            '  pip install "lerobot[dataset]>=0.4,<0.7"\n'
            f'Original import error: {exc}'
        ) from exc
    return LeRobotDataset, RGBEncoderConfig


def _init_worker(config: WorkerConfig) -> None:
    global _WORKER_CONFIG, _WORKER_ENV

    # Import after process creation: MuJoCo/OpenGL contexts must not be inherited.
    import ocbench

    _WORKER_CONFIG = config
    control_timestep = 1.0 / config.fps
    _WORKER_ENV = ocbench.make(
        ENV_NAME,
        control_timestep=control_timestep,
        physics_timestep=control_timestep / config.physics_substeps,
        width=config.width,
        height=config.height,
    )


def _rollout_episode(episode_index: int, seed: int) -> dict:
    from ocbench.oracles.controllers.hanoi import HanoiController

    assert _WORKER_CONFIG is not None and _WORKER_ENV is not None
    config = _WORKER_CONFIG
    env = _WORKER_ENV
    last_status = ''

    for attempt in range(config.max_attempts):
        attempt_seed = (seed + attempt * 1_000_003) % (2**32)
        np.random.seed(attempt_seed)
        observation, info = env.reset(seed=attempt_seed)
        controller = HanoiController(env=env)
        controller.reset(observation, info, transition_idx=0)

        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        successes: list[np.ndarray] = []
        images: dict[str, list[np.ndarray]] = {camera: [] for camera in config.cameras}
        episode_success = False
        episode_healthy = True

        for step in range(config.max_steps):
            action = np.asarray(controller.select_action(observation, info, transition_idx=step), dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)

            states.append(np.asarray(observation, dtype=np.float32))
            actions.append(action)
            for camera in config.cameras:
                render_camera = CAMERA_RENDER_NAMES.get(camera, camera)
                images[camera].append(np.asarray(env.unwrapped.render(camera=render_camera), dtype=np.uint8))

            next_observation, reward, terminated, truncated, info = env.step(action)
            timed_out = step + 1 >= config.max_steps
            done = bool(terminated or truncated or timed_out)
            episode_success = episode_success or bool(info['success'])
            episode_healthy = episode_healthy and bool(info['healthy'])
            rewards.append(np.asarray([reward], dtype=np.float32))
            dones.append(np.asarray([done], dtype=np.bool_))
            successes.append(np.asarray([info['success']], dtype=np.bool_))
            observation = next_observation
            if done:
                break

        accepted = episode_healthy and (episode_success or not config.successful_only)
        if accepted:
            payload = {
                'observation.state': np.stack(states),
                'action': np.stack(actions),
                'next.reward': np.stack(rewards),
                'next.done': np.stack(dones),
                'next.success': np.stack(successes),
            }
            for camera, frames in images.items():
                payload[f'observation.images.{camera}'] = np.stack(frames)
            episode_path = Path(config.temp_dir) / f'episode_{episode_index:08d}.npz'
            np.savez(episode_path, **payload)
            return {
                'episode_index': episode_index,
                'path': str(episode_path),
                'length': len(actions),
                'success': episode_success,
                'attempts': attempt + 1,
            }

        last_status = (
            f'success={episode_success}, healthy={episode_healthy}, length={len(actions)}, seed={attempt_seed}'
        )

    raise RuntimeError(
        f'Episode {episode_index} was not accepted after {config.max_attempts} attempts ({last_status}). '
        'Increase --max-attempts, use --no-successful-only, or inspect the oracle/environment.'
    )


def _episode_jobs(num_episodes: int, root_seed: int) -> list[tuple[int, int]]:
    seed_sequence = np.random.SeedSequence(root_seed)
    child_sequences = seed_sequence.spawn(num_episodes)
    return [
        (episode_index, int(child.generate_state(1, dtype=np.uint32)[0]))
        for episode_index, child in enumerate(child_sequences)
    ]


def _generate_episodes(config: WorkerConfig, jobs: list[tuple[int, int]], workers: int) -> Iterator[dict]:
    if workers == 1:
        _init_worker(config)
        try:
            for job in jobs:
                yield _rollout_episode(*job)
        finally:
            _WORKER_ENV.close()
        return

    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(config,),
    ) as executor:
        futures = [executor.submit(_rollout_episode, *job) for job in jobs]
        try:
            for future in futures:
                yield future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _features(sample: np.lib.npyio.NpzFile, cameras: Sequence[str]) -> dict:
    features = {
        'observation.state': {
            'dtype': 'float32',
            'shape': sample['observation.state'].shape[1:],
            'names': None,
        },
        'action': {'dtype': 'float32', 'shape': sample['action'].shape[1:], 'names': None},
        'next.reward': {'dtype': 'float32', 'shape': (1,), 'names': ['reward']},
        'next.done': {'dtype': 'bool', 'shape': (1,), 'names': ['done']},
        'next.success': {'dtype': 'bool', 'shape': (1,), 'names': ['success']},
    }
    for camera in cameras:
        key = f'observation.images.{camera}'
        height, width, channels = sample[key].shape[1:]
        features[key] = {
            'dtype': 'video',
            'shape': (height, width, channels),
            'names': ['height', 'width', 'channel'],
        }
    return features


def _validate_video_dataset(output_dir: Path, cameras: Sequence[str], fps: int) -> None:
    info_path = output_dir / 'meta' / 'info.json'
    with info_path.open() as file:
        info = json.load(file)

    if info.get('codebase_version') != 'v3.0':
        raise RuntimeError(f'Expected LeRobot v3.0 metadata in {info_path}.')
    if info.get('fps') != fps:
        raise RuntimeError(f'Expected {fps} FPS in {info_path}, got {info.get("fps")}.')

    for camera in cameras:
        key = f'observation.images.{camera}'
        feature = info['features'].get(key)
        if feature is None or feature.get('dtype') != 'video':
            raise RuntimeError(f'{key} is not declared as a video feature in {info_path}.')
        video_info = feature.get('info', {})
        if video_info.get('video.codec') != 'h264' or video_info.get('video.pix_fmt') != 'yuv420p':
            raise RuntimeError(f'{key} was not encoded as H.264/yuv420p: {video_info}')
        video_dir = output_dir / 'videos' / key
        if not any(video_dir.rglob('*.mp4')):
            raise RuntimeError(f'No MP4 files were written for {key} under {video_dir}.')


def _append_episode(dataset, result: dict) -> None:
    episode_path = Path(result['path'])
    with np.load(episode_path, mmap_mode='r', allow_pickle=False) as episode:
        frame_keys = list(episode.files)
        length = episode[frame_keys[0]].shape[0]
        for frame_index in range(length):
            frame = {key: episode[key][frame_index] for key in frame_keys}
            frame['task'] = TASK_DESCRIPTION
            dataset.add_frame(frame)
        dataset.save_episode()
    episode_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    LeRobotDataset, RGBEncoderConfig = _require_lerobot()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f'Output directory already exists: {output_dir}\nUse --overwrite to replace it.')
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='.hanoi_lerobot_', dir=output_dir.parent) as temp_dir:
        config = WorkerConfig(
            temp_dir=temp_dir,
            fps=args.fps,
            physics_substeps=args.physics_substeps,
            max_steps=args.max_steps,
            cameras=tuple(args.cameras),
            width=args.width,
            height=args.height,
            successful_only=args.successful_only,
            max_attempts=args.max_attempts,
        )
        jobs = _episode_jobs(args.num_episodes, args.seed)
        results = _generate_episodes(config, jobs, min(args.workers, args.num_episodes))
        completed = 0
        total_frames = 0
        with tqdm(
            total=args.num_episodes,
            desc='Generating LeRobot dataset',
            unit='episode',
            dynamic_ncols=True,
        ) as progress:
            first_result = next(results)

            with np.load(first_result['path'], mmap_mode='r', allow_pickle=False) as sample:
                features = _features(sample, args.cameras)
            encoder_preset = 'p1' if args.video_encoder == 'h264_nvenc' else 'veryfast'
            rgb_encoder = RGBEncoderConfig(
                vcodec=args.video_encoder,
                pix_fmt='yuv420p',
                g=args.fps,
                crf=23,
                preset=encoder_preset,
            )
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                fps=args.fps,
                root=output_dir,
                robot_type='ocbench_ur5e_robotiq_2f85',
                features=features,
                use_videos=True,
                image_writer_threads=args.image_writer_threads,
                rgb_encoder=rgb_encoder,
            )

            try:
                for result in itertools.chain((first_result,), results):
                    _append_episode(dataset, result)
                    completed += 1
                    total_frames += result['length']
                    progress.update()
                    progress.set_postfix(
                        episode=result['episode_index'],
                        frames=result['length'],
                        attempts=result['attempts'],
                    )
            finally:
                dataset.finalize()

    _validate_video_dataset(output_dir, args.cameras, args.fps)
    print(f'Created {output_dir} ({completed} episodes, {total_frames} frames, {args.fps} Hz).')
    if args.push_to_hub:
        dataset.push_to_hub(private=args.private)
        print(f'Pushed {args.repo_id} to the Hugging Face Hub.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
