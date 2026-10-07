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
  --video-encoder h264 \
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

Only the parent process writes the LeRobot dataset. Workers independently
simulate episodes and stage them temporarily, which prevents concurrent
Parquet/video metadata corruption.
