#!/bin/bash
#SBATCH --job-name=dreamer_reacher
#SBATCH --account=def-rsdjjana
#SBATCH --time=6-23:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0
#SBATCH --acctg-freq=task=1
#SBATCH --output=/home/ruism/projects/def-rsdjjana/ruism/Dreamer/original/%A-%a.out
#SBATCH --error=/home/ruism/projects/def-rsdjjana/ruism/Dreamer/original/%A-%a.err

BASE_SAVE_DIR="$HOME/projects/def-rsdjjana/ruism/Dreamer/original/reacherloca"

# Make sure log dir exists (SLURM will drop stdout/err here)
mkdir -p "$BASE_SAVE_DIR"
# Gentle stagger so all tasks don’t hammer Lustre at once
sleep $(( (SLURM_ARRAY_TASK_ID % 10) * 3 ))

# --- Clean env, load modules ---
set -e -o pipefail
module --force purge
set +u
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
set -u
module load StdEnv/2020
module load cuda/11.4
module load glfw/3.3.2

module load ffmpeg/4.3.2
export IMAGEIO_FFMPEG_EXE="$(command -v ffmpeg)"

# --- Activate your venv ---
source "$HOME/projects/def-rsdjjana/ruism/loca_env/bin/activate"

# --- MuJoCo 2.1.0 (legacy) runtime bits ---
export MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export MUJOCO_PLUGIN_PATH="$MUJOCO_PATH/bin/mujoco_plugin"
export MUJOCO_GL=glfw
export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:$EBROOTGLFW/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

# --- Threading + wandb ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WANDB_MODE=offline

# --- Per-task scratch run dir and final dir ---
: "${SLURM_TMPDIR:=/tmp}"
RUN_DIR="${SLURM_TMPDIR}/dreamer-${SLURM_JOB_ID:-0}-${SLURM_ARRAY_TASK_ID:-0}"
FINAL_DIR="${BASE_SAVE_DIR}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN_DIR" "$FINAL_DIR"

# --- On exit, rsync scratch → final (even on failure) ---
_finish() {
  if [ -d "$RUN_DIR" ]; then
    rsync -a --partial --inplace --no-whole-file "$RUN_DIR/" "$FINAL_DIR/"
  fi
}
trap _finish EXIT TERM INT

# --- Where the source code lives ---
DREAMER_SRC="$HOME/projects/def-rsdjjana/ruism/Dreamer"

# Run *from scratch* so all outputs land in $RUN_DIR
cd "$RUN_DIR"

# Ensure local imports (env_wrapper.py, models.py, etc.) resolve
export PYTHONPATH="$DREAMER_SRC:${PYTHONPATH:-}"

# (Optional) CUDA probe in logs
python - <<'PY'
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch probe failed:", e)
PY

# --- Train (seed = array index) ---
python -u "$DREAMER_SRC/dreamer.py" \
  --env 'reacherloca' \
  --algo 'Dreamerv2' \
  --exp 'default_hp' \
  --train \
  --seed "${SLURM_ARRAY_TASK_ID}"
