#!/bin/bash
#SBATCH -o logs/out/%j.out
#SBATCH -e logs/err/%j.err
#SBATCH --partition ccn
#SBATCH --mail-type=FAIL
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4

start_time=$(date +%s)
mode="linear_probe"
num_process=5

module -q purge
module -q load openmpi
mpirun -n $num_process python dnn_pipeline.py \
    -p "params/params_test.json" \
    -m $mode \

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job Duration: $(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))"