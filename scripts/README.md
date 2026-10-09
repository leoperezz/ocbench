# LeRobot data generation

`generate_hanoi_lerobot.py` records oracle demonstrations for the three-disk,
random-layout Hanoi task (`hanoi-cpu-triple-task2-v0`) in LeRobot v3 video
format. The simulation and the resulting H.264 videos both run at 30 Hz by
default.

Install the optional dependencies:

```bash
pip install -e ".[data]"
```

Record the front, side, and wrist RGB streams as LeRobot videos:

```bash
MUJOCO_GL=egl python scripts/generate_hanoi_lerobot.py \
  --num-episodes 525 \
  --workers 8 \
  --cameras front side wrist \
  --video \
  --video-encoder h264_nvenc \
  --fps 30 \
  --output-dir data/lerobot/hanoi-triple-task2-rgb
```

The camera feature names follow the LeRobot convention
`observation.images.<camera>`, but their storage type is `video`; final pixels
are stored under `videos/observation.images.<camera>/**/*.mp4`, not as image
files. H.264, `yuv420p`, and an HWC feature schema are used for compatibility
with the previous OCBench-to-LeRobot converter.

Useful options include `--seed`, `--fps`, `--max-steps`,
`--video-encoder`, `--[no-]successful-only`, `--max-attempts`, `--overwrite`, and
`--push-to-hub`. Run the script with `--help` for the complete interface.

`observation.state` and `action` both contain exactly seven proprioceptive robot
values. The first six are absolute arm joint positions in radians and the last
is gripper opening in `[0, 1]`. The state contains the current positions and the
action contains the absolute actuator targets. Object poses, joint velocities,
end-effector state, and contact state are not stored in `observation.state`.

The oracle still receives OCBench's complete internal observation and rollouts
still execute the environment's normalized delta control. Those details are
used only to generate the demonstration and are converted before being written
to the dataset.

When a model trained on absolute actions is evaluated in OCBench, convert its
prediction back to the environment action before calling `env.step()`:

```python
scale = np.array([0.18, 0.18, 0.18, 0.36, 0.36, 0.36, 0.12])
joint_state = np.concatenate([env_observation[:6], [env_observation[17] / 3.0]])
predicted_target = model(joint_state, images)
env_action = np.clip((predicted_target - joint_state) / scale, -1.0, 1.0)
env_observation, reward, terminated, truncated, info = env.step(env_action)
```

Rollouts run in parallel. At most one job per worker is in flight, and a new job
is submitted only after a completed `.npz` has been appended to LeRobot and
deleted. Consequently, temporary rollout storage is bounded by `--workers`
episodes rather than growing with the total dataset. Streaming video encoding
is enabled by default, avoiding temporary PNG frame directories as well. The
final dataset itself naturally grows as episodes are added.
