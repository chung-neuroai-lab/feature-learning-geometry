## Overview

The pipeline is created to train DNNs models (ResNet, VGG) with different hyperparmeters (scaling parameters, learning rate) to induce rich vs lazy learning conditions (see [On Lazy Training in Differentiable Programming](https://proceedings.neurips.cc/paper_files/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf)) and measure different feature learning metrics including weight-based metrics (weight change), activation-based metrics (manifold capacity, activation stability, representation-label alignment), kernel-based metrics (NTK change, NTK-label alignment).

## Pipeline structure

The pipeline has 3 main steps:
1. Model Training: Train the DNN models
2. Feature Extraction: Extract feature representations from the trained models
3. Feature analysis: Analyze extracted features using various metrics, including weight-based, activation-based, and kernel-based metrics.

## Step-by-step guide to run the pipeline

1. Set up `config.json`
    1. Open config.json, set `filename_config[input_data]` to the folder path which has the dataset (CIFAR-10 and CIFAR-100) and `filename_config[corrupt_data_folder]` to the folder path which has the corrupted dataset (CIFAR-10C)
2. Define input parameters:
    1. Run `generate_input_params.ipynb` to specify the input parameters combination to be used in the pipeline. This notebook will create a file `params_{timestamp}.json` that contains the list of parameters to be used in the pipeline.
3. Run the pipeline: The pipeline can be ran via `slurm_job_gpu_mpi.sh` (use GPU and usually used for model training) and `slurm_job_cpu_mpi.sh` (use CPU and usually used for feature extraction and analysis)
    1. In `slurm_job_gpu_mpi.sh` and `slurm_job_cpu_mpi.sh`, change `-p` to the location of the `params_{timestamp}.jso`n file above and `-m` to the mode that we want to run (supported mode is in `config.jso`n)
    2. Activate the Python virtual environment, submit job via `sbatch --array=0-{len_array} slurm_job_gpu_mpi.sh` or `sbatch --array=0-{len_array} slurm_job_gpu_mpi.sh` (`len_array` is available in `generate_input_params.ipynb` when generating the `params_{timestamp}.json`)
