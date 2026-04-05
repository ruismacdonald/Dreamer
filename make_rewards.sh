#!/bin/bash
#SBATCH --job-name=rewards
#SBATCH --account=def-rsdjjana
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=ng[11105,30708]
#SBATCH --mem=16G
#SBATCH --array=1
#SBATCH --acctg-freq=task=1
#SBATCH --output=/home/ruism/projects/def-rsdjjana/ruism/Dreamer/results/rewards/%A-%a.out
#SBATCH --error=/home/ruism/projects/def-rsdjjana/ruism/Dreamer/results/rewards/%A-%a.err

set -e -o pipefail

DATA_BASE="$HOME/projects/def-rsdjjana/ruism/Dreamer"

# Gentle stagger so all tasks don’t hammer Lustre at once
sleep $(( (SLURM_ARRAY_TASK_ID % 10) * 3 ))

# Modules/enviroment
module --force purge
set +u
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
set -u
module load StdEnv/2020
module load cuda/11.4
module load glfw/3.3.2
module load ffmpeg/4.3.2
export IMAGEIO_FFMPEG_EXE="$(command -v ffmpeg)"

# Activate venv
source "$HOME/projects/def-rsdjjana/ruism/loca_env/bin/activate"

# MuJoCo 2.1.0 (legacy)
export MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export MUJOCO_PLUGIN_PATH="$MUJOCO_PATH/bin/mujoco_plugin"
export MUJOCO_GL=glfw
export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:$EBROOTGLFW/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

# Threading + wandb
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WANDB_MODE=offline

# Keep BLAS backends from spawning their own pools (most common crash source)
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Make OpenMP behavior stable under Slurm binding
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Consistent CPU binding
export SLURM_CPU_BIND=cores

DREAMER_SRC="$HOME/projects/def-rsdjjana/ruism/Dreamer"
export PYTHONPATH="$DREAMER_SRC:${PYTHONPATH:-}"

export LOCA_DATALOADER_WORKERS=0


: "${SLURM_TMPDIR:=/tmp}"
SEED="${SLURM_ARRAY_TASK_ID}"



CKPT_PATH="${DATA_BASE}/data/reacherloca_fifo_2/${SEED}/phase_1/ckpts/models.pt"

OUT_BASE="$DREAMER_SRC/data/rewards/fifo_${SEED}_phase1"
mkdir -p "$OUT_BASE"

RUN_DIR="${SLURM_TMPDIR}/dreamer-rewards-${SLURM_JOB_ID:-0}-${SEED}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

python -u "$DREAMER_SRC/make_rewards.py" \
  --env 'reacherloca-easy' \
  --algo 'Dreamerv2' \
  --exp-name 'reacherloca_fifo_rewards' \
  --seed "${SEED}" \
  --action-repeat 2 \
  --checkpoint-model-path "$CKPT_PATH"

rsync -a "$RUN_DIR/heat_data.npy" "$OUT_BASE/"
echo "Saved to: $OUT_BASE/heat_data.npy"



CKPT_PATH="${DATA_BASE}/data/reacherloca_v1_rad_03/${SEED}/phase_1/ckpts/models.pt"

OUT_BASE="$DREAMER_SRC/data/rewards/v1_${SEED}_phase1"
mkdir -p "$OUT_BASE"

RUN_DIR="${SLURM_TMPDIR}/dreamer-rewards-${SLURM_JOB_ID:-0}-${SEED}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

python -u "$DREAMER_SRC/make_rewards.py" \
  --env 'reacherloca-easy' \
  --algo 'Dreamerv2' \
  --exp-name 'reacherloca_v1_rewards' \
  --seed "${SEED}" \
  --action-repeat 2 \
  --checkpoint-model-path "$CKPT_PATH"

rsync -a "$RUN_DIR/heat_data.npy" "$OUT_BASE/"
echo "Saved to: $OUT_BASE/heat_data.npy"



CKPT_PATH="${DATA_BASE}/../Dreamer/data/reacherloca_v2_norm_64_500/${SEED}/phase_1/ckpts/models.pt"

OUT_BASE="$DREAMER_SRC/data/rewards/v2_${SEED}_phase1"
mkdir -p "$OUT_BASE"

RUN_DIR="${SLURM_TMPDIR}/dreamer-rewards-${SLURM_JOB_ID:-0}-${SEED}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

python -u "$DREAMER_SRC/make_rewards.py" \
  --env 'reacherloca-easy' \
  --algo 'Dreamerv2' \
  --exp-name 'reacherloca_v2_rewards' \
  --seed "${SEED}" \
  --action-repeat 2 \
  --checkpoint-model-path "$CKPT_PATH"

rsync -a "$RUN_DIR/heat_data.npy" "$OUT_BASE/"
echo "Saved to: $OUT_BASE/heat_data.npy"



CKPT_PATH="${DATA_BASE}/data/reacherloca_fifo_2/${SEED}/phase_2/ckpts/models.pt"

OUT_BASE="$DREAMER_SRC/data/rewards/fifo_${SEED}_phase2"
mkdir -p "$OUT_BASE"

RUN_DIR="${SLURM_TMPDIR}/dreamer-rewards-${SLURM_JOB_ID:-0}-${SEED}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

python -u "$DREAMER_SRC/make_rewards.py" \
  --env 'reacherloca-easy' \
  --algo 'Dreamerv2' \
  --exp-name 'reacherloca_fifo_rewards' \
  --seed "${SEED}" \
  --action-repeat 2 \
  --checkpoint-model-path "$CKPT_PATH"

rsync -a "$RUN_DIR/heat_data.npy" "$OUT_BASE/"
echo "Saved to: $OUT_BASE/heat_data.npy"



CKPT_PATH="${DATA_BASE}/data/reacherloca_v1_rad_03/${SEED}/phase_2/ckpts/models.pt"

OUT_BASE="$DREAMER_SRC/data/rewards/v1_${SEED}_phase2"
mkdir -p "$OUT_BASE"

RUN_DIR="${SLURM_TMPDIR}/dreamer-rewards-${SLURM_JOB_ID:-0}-${SEED}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

python -u "$DREAMER_SRC/make_rewards.py" \
  --env 'reacherloca-easy' \
  --algo 'Dreamerv2' \
  --exp-name 'reacherloca_v1_rewards' \
  --seed "${SEED}" \
  --action-repeat 2 \
  --checkpoint-model-path "$CKPT_PATH"

rsync -a "$RUN_DIR/heat_data.npy" "$OUT_BASE/"
echo "Saved to: $OUT_BASE/heat_data.npy"



CKPT_PATH="${DATA_BASE}/../Dreamer/data/reacherloca_v2_norm_64_500/${SEED}/phase_2/ckpts/models.pt"

OUT_BASE="$DREAMER_SRC/data/rewards/v2_${SEED}_phase2"
mkdir -p "$OUT_BASE"

RUN_DIR="${SLURM_TMPDIR}/dreamer-rewards-${SLURM_JOB_ID:-0}-${SEED}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

python -u "$DREAMER_SRC/make_rewards.py" \
  --env 'reacherloca-easy' \
  --algo 'Dreamerv2' \
  --exp-name 'reacherloca_v2_rewards' \
  --seed "${SEED}" \
  --action-repeat 2 \
  --checkpoint-model-path "$CKPT_PATH"

rsync -a "$RUN_DIR/heat_data.npy" "$OUT_BASE/"
echo "Saved to: $OUT_BASE/heat_data.npy"



CKPT_PATH="${DATA_BASE}/data/reacherloca_fifo_rew/${SEED}/phase_2/ckpts/models.pt"

OUT_BASE="$DREAMER_SRC/data/rewards_rew/fifo_${SEED}_phase2"
mkdir -p "$OUT_BASE"

RUN_DIR="${SLURM_TMPDIR}/dreamer-rewards-rew-${SLURM_JOB_ID:-0}-${SEED}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

python -u "$DREAMER_SRC/make_rewards.py" \
  --env 'reacherloca-easy' \
  --algo 'Dreamerv2' \
  --exp-name 'reacherloca_fifo_rewards_rew' \
  --seed "${SEED}" \
  --action-repeat 2 \
  --checkpoint-model-path "$CKPT_PATH"

rsync -a "$RUN_DIR/heat_data.npy" "$OUT_BASE/"
echo "Saved to: $OUT_BASE/heat_data.npy"



CKPT_PATH="${DATA_BASE}/data/reacherloca_v1_rad_03_rew/${SEED}/phase_2/ckpts/models.pt"

OUT_BASE="$DREAMER_SRC/data/rewards_rew/v1_${SEED}_phase2"
mkdir -p "$OUT_BASE"

RUN_DIR="${SLURM_TMPDIR}/dreamer-rewards-rew-${SLURM_JOB_ID:-0}-${SEED}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

python -u "$DREAMER_SRC/make_rewards.py" \
  --env 'reacherloca-easy' \
  --algo 'Dreamerv2' \
  --exp-name 'reacherloca_v1_rewards_rew' \
  --seed "${SEED}" \
  --action-repeat 2 \
  --checkpoint-model-path "$CKPT_PATH"

rsync -a "$RUN_DIR/heat_data.npy" "$OUT_BASE/"
echo "Saved to: $OUT_BASE/heat_data.npy"



CKPT_PATH="${DATA_BASE}/../Dreamer/data/reacherloca_v2_norm_64_500_rew/${SEED}/phase_2/ckpts/models.pt"

OUT_BASE="$DREAMER_SRC/data/rewards_rew/v2_${SEED}_phase2"
mkdir -p "$OUT_BASE"

RUN_DIR="${SLURM_TMPDIR}/dreamer-rewards-rew-${SLURM_JOB_ID:-0}-${SEED}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

python -u "$DREAMER_SRC/make_rewards.py" \
  --env 'reacherloca-easy' \
  --algo 'Dreamerv2' \
  --exp-name 'reacherloca_v2_rewards_rew' \
  --seed "${SEED}" \
  --action-repeat 2 \
  --checkpoint-model-path "$CKPT_PATH"

rsync -a "$RUN_DIR/heat_data.npy" "$OUT_BASE/"
echo "Saved to: $OUT_BASE/heat_data.npy"