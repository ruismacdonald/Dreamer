#!/bin/bash
#SBATCH --job-name=plot_reacherloca
#SBATCH --account=def-rsdjjana
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=results/reacherloca/plot_reacherloca%j.out
#SBATCH --error=results/reacherloca/plot_reacherloca%j.err

module --force purge
set +u
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
set -u

module load StdEnv/2020
module load glfw/3.3.2
source ~/projects/def-rsdjjana/ruism/loca_env/bin/activate

cd ~/projects/def-rsdjjana/ruism/Dreamer

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python plot_reacherloca.py \
  --indir  data/reacherloca_fifo/ data/reacherloca_v1_rad_1  data/reacherloca_v1_rad_2  data/reacherloca_v2_norm_100  data/reacherloca_v2_norm_50 \
  --outdir results/reacherloca \
  --subdir False \
  --xaxis step \
  --yaxis eval_avg_reward \
  --bins 40000 \
  --methods reacherloca_fifo reacherloca_v1_rad_1  reacherloca_v1_rad_2  reacherloca_v2_norm_100  reacherloca_v2_norm_50 \
  --add none \
  --labels \
    reacherloca_fifo "DreamerV2 + FIFO" \
    reacherloca_v1_rad_1 "DreamerV2 + LoFoV1 (rad 1)" \
    reacherloca_v1_rad_2 "DreamerV2 + LoFoV1 (rad 2)" \
    reacherloca_v2_norm_100 "DreamerV2 + LoFoV2 (norm 100)" \
    reacherloca_v2_norm_50 "DreamerV2 + LoFoV2 (norm 50)"

# data/reacherloca_v1_2/ data/reacherloca_v1_5/ data/reacherloca_v1_10/ data/reacherloca_v2/
#reacherloca_v1_2  reacherloca_v1_5  reacherloca_v1_10  reacherloca_v2
# reacherloca_v1_2 "DreamerV2 + LoFoV1 (2)" \
# reacherloca_v1_5 "DreamerV2 + LoFoV1 (5)" \
# reacherloca_v1_10 "DreamerV2 + LoFoV1 (10)" \
# reacherloca_v2 "DreamerV2 + LoFoV2" \

# python plot_reacherloca.py \
#   --indir  data/reacherloca_fifo_10seeds_save/ data/reacherloca_v1_10seeds_save/ ../Dreamer_v2_SDH/data/reacherloca_v2_state_dist_10_seeds/ ../Dreamer_v2_SDH/data/reacherloca_v2_rep_norm_state_dist_10_seeds/ \
#   --outdir results/reacherloca/r \
#   --subdir False \
#   --xaxis step \
#   --yaxis eval_avg_reward \
#   --bins 40000 \
#   --methods reacherloca_fifo_10seeds_save reacherloca_v1_10seeds_save reacherloca_v2_state_dist_10_seeds reacherloca_v2_rep_norm_state_dist_10_seeds\
#   --add none \
#   --labels \
#     reacherloca_fifo_10seeds_save "DreamerV2 + FIFO" \
#     reacherloca_v1_10seeds_save "DreamerV2 + LoFoV1" \
#     reacherloca_v2_state_dist_10_seeds "DreamerV2 + LoFoV2 (no norm)" \
#     reacherloca_v2_rep_norm_state_dist_10_seeds "DreamerV2 + LoFoV2"

# python plot_randomizedreacherloca.py \
#   --indir  data/randomizedreacherloca_fifo_10seeds/ data/randomizedreacherloca_v1_10seeds/ ../Dreamer_v2_SDH/data/randomizedreacherloca_v2_rep_norm_state_dist_10_seeds/ \
#   --outdir results/reacherloca/rr \
#   --subdir False \
#   --xaxis step \
#   --yaxis eval_avg_reward \
#   --bins 40000 \
#   --methods randomizedreacherloca_fifo_10seeds randomizedreacherloca_v1_10seeds randomizedreacherloca_v2_rep_norm_state_dist_10_seeds \
#   --add none \
#   --labels \
#     randomizedreacherloca_fifo_10seeds "DreamerV2 + FIFO" \
#     randomizedreacherloca_v1_10seeds "DreamerV2 + LoFoV1" \
#     randomizedreacherloca_v2_rep_norm_state_dist_10_seeds "DreamerV2 + LoFoV2 (norm)"
    