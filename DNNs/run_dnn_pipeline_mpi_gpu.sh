#!/bin/bash
#SBATCH -o logs/out/%j.out
#SBATCH -e logs/err/%j.err
#SBATCH --mail-type=FAIL
#SBATCH --time=24:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1

start_time=$(date +%s)
mode="model_training"

module -q purge
module -q load openmpi
module -q load cuda

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --cpu-bind=cores --gpu-bind=single:2 \
    -c $SLURM_CPUS_PER_TASK \
    python dnn_pipeline.py \
    -p "params/params_test.json" \
    -m $mode \

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job Duration: $(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))"