# LeRobot data generation

`generate_hanoi_lerobot.py` records oracle demonstrations for the three-disk,
random-layout Hanoi task (`hanoi-cpu-triple-task2-v0`) in LeRobot v3 format.
The simulation and the resulting dataset both run at 30 Hz by default.

Install the optional dependencies:

```bash
pip install -e ".[data]"
```

Generate 100 state-only episodes with 8 parallel simulation workers:

```bash
python scripts/generate_hanoi_lerobot.py \
  --num-episodes 100 \
  --workers 8 \
  --output-dir data/lerobot/hanoi-triple-task2
```

Record front and wrist RGB observations as LeRobot videos too:

```bash
MUJOCO_GL=egl python scripts/generate_hanoi_lerobot.py \
  --num-episodes 100 \
  --workers 8 \
  --cameras front wrist \
  --video \
  --output-dir data/lerobot/hanoi-triple-task2-rgb
```

Useful options include `--seed`, `--fps`, `--max-steps`,
`--[no-]successful-only`, `--max-attempts`, `--overwrite`, and
`--push-to-hub`. Run the script with `--help` for the complete interface.

Only the parent process writes the LeRobot dataset. Workers independently
simulate episodes and stage them temporarily, which prevents concurrent
Parquet/video metadata corruption.
