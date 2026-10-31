# LeRobot data generation

`generate_hanoi_lerobot.py` records oracle demonstrations for the random-layout
Hanoi task (`hanoi-cpu-triple-task2-v0`) in LeRobot v3 video
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
  --with-reward \
  --with-done \
  --output-dir data/lerobot/hanoi-triple-task2-rgb
```

The camera feature names follow the LeRobot convention
`observation.images.<camera>`, but their storage type is `video`; final pixels
are stored under `videos/observation.images.<camera>/**/*.mp4`, not as image
files. H.264, `yuv420p`, and an HWC feature schema are used for compatibility
with the previous OCBench-to-LeRobot converter.

Use `--num-disks 1`, `--num-disks 2`, or `--num-disks 3` (default) to choose
how many physical disks are created, even with a `triple` environment ID.
Three is the current peg capacity. The same option works for disk-in-tower
and autoreset; task text and subtask labels adjust to the count. Two disks
are red and blue; one disk is blue. Scattered layouts use the first N floor
positions (left, center, right). With one disk, autoreset resamples its yaw
at the same floor position. For direct environment use, pass `num_disks=2`
to `ocbench.make(...)`.

For example, to collect two-disk stacking demonstrations:

```bash
MUJOCO_GL=egl python scripts/generate_hanoi_lerobot.py \
  --env-name hanoi-triple-disk-in-tower-v0 \
  --num-disks 2 --num-episodes 100 --subtask \
  --output-dir data/lerobot/hanoi-two-disk-stack
```

The CPU oracle orients the gripper according to the grasp side, putting the
hole toward the wrist camera during insertion. The generator also checks
visibility on each insertion frame: at least three of the four inner rim
corners must be in the wrist image and unobstructed according to MuJoCo ray
casts. An attempt that fails this check is discarded in full and retried up
to `--max-attempts`, even with `--no-successful-only` or without recording the
wrist stream. This check covers the insertion descent, not the entire episode
(e.g. the empty gripper's approach or retreat). Occlusions can therefore
increase the number of attempts needed.

Useful options include `--num-disks`, `--seed`, `--fps`, `--max-steps`, `--with-reward`, `--with-done`,
`--video-encoder`, `--[no-]successful-only`, `--max-attempts`, `--overwrite`, and
`--push-to-hub`. Run the script with `--help` for the complete interface.

`--with-reward` adds the standard LeRobot `next.reward` feature as `float32`,
using the reward returned by the environment for each transition. `--with-done`
adds `next.done` as a boolean; it is true when that transition terminates,
truncates, or reaches the configured generator time limit. Both features and
their types are included in `meta/info.json` when enabled.

For `hanoi-triple-disk-in-tower-v0`, pass `--subtask` to add a
frame-level `subtask_index` feature and `meta/subtasks.parquet`. The subtask
labels follow the oracle's execution order: with three disks, place the blue
disk, then the orange disk (yellow in autoreset), then the red disk. Each label includes the frame that completes
that placement; the next frame starts the following subtask. Other environments
report that this flag is not supported.

The generator maps this public environment ID to its single-world CPU backend
internally, so `-cpu-` is not needed in the command.

The autoreset variant records the arm's reset trajectory as well:

```bash
MUJOCO_GL=egl python scripts/generate_hanoi_lerobot.py \
  --env-name hanoi-triple-disk-in-tower-autoreset-v0 \
  --num-episodes 3 \
  --workers 8 \
  --max-steps 14000 \
  --subtask \
  --output-dir data/lerobot/hanoi-triple-disk-in-tower-autoreset-rgb
```

Frames use the stacking task instruction until the tower is complete and the
reset instruction afterward. Each episode executes N placements onto the
tower and N placements into a newly sampled floor configuration. The subtask
mapping contains N + N² labels: N for stacking and N² for every possible
disk/floor-position pair (twelve labels with the default three disks). Parallel collection is supported: every worker owns
one persistent environment and continues from its own previous reset endpoint.
The resulting worker chains are independent, while all their reset endpoints
belong to the same valid initial-state distribution.

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
