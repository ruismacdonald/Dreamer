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
  --indir  data/reacherloca_fifo/ data/reacherloca_v1_rad_03 data/reacherloca_v1_rad_04  data/reacherloca_v2_norm_64_500  data/reacherloca_v2_norm_256_100 \
  --outdir results/reacherloca \
  --subdir False \
  --xaxis step \
  --yaxis eval_avg_reward \
  --bins 40000 \
  --methods reacherloca_fifo reacherloca_v1_rad_03  reacherloca_v1_rad_04  reacherloca_v2_norm_64_500  reacherloca_v2_norm_256_100 \
  --add none \
  --labels \
    reacherloca_fifo "DreamerV2 + FIFO" \
    reacherloca_v1_rad_03 "DreamerV2 + LoFoV1 (rad 0.3)" \
    reacherloca_v1_rad_04 "DreamerV2 + LoFoV1 (rad 0.4)" \
    reacherloca_v2_norm_64_500 "DreamerV2 + LoFoV2 (norm 64, 500)" \
    reacherloca_v2_norm_256_100 "DreamerV2 + LoFoV2 (norm 256, 100)" 
   
    