def main(params_filename, mode):
    """Assign the parameter to the current calling process.
    Given a `.json` file at the path `params_filename`, use `mpi4py` to return the number of total processes,
    the rank of the current calling process, and the environment variable `SLURM_ARRAY_TASK_ID`
    to determine the index of the parameter to be used in the list of parameters in the `.json` file.

    Args:
        params_filename (str): Path to the parameter `.json` file
        mode (str): Pipeline mode. See `config.json` for the supported modes.
    """   
    from mpi4py import MPI
    import os
    import sys
    import json
    import run_train_model_lazy, run_feature_analysis, run_linear_probe, run_eval_corrupt, run_kernel_methods, run_train_model_lazy_step

    # read MPI config
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_threads = comm.Get_size()
    print(f"Running as MPI thread {rank} ({num_threads} total)")

    # read Slurm config
    job_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    num_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    print(f'running as job {job_id}, task {task_id} ({num_tasks} total)')

    # read params file
    pipeline_params = json.load(open(params_filename, "r"))
    params = pipeline_params[mode]

    # get parameter for this job (in the parameter list json file)
    job_idx = rank + (num_threads * task_id)
    if job_idx > len(params) - 1:
        print(f"job index {job_idx} is out of range for params file with length {len(params)} at path {params_filename}")
        sys.exit(0)
    
    # run Python function depends on the param mode
    this_param = params[job_idx]
    this_param["mode"] = mode
    mode_to_func = {
        "model_training": run_train_model_lazy_step,
        # "model_training": run_train_model_lazy,
        "feature_analysis": run_feature_analysis,
        "linear_probe": run_linear_probe,
        "eval_corrupt": run_eval_corrupt,
        "kernels": run_kernel_methods,
    }
    if mode not in mode_to_func:
        raise ValueError(f"Mode {mode} is in list of supported mode {mode_to_func.keys()}")
    else:
        mode_to_func[mode].main(this_param)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--params_filename', required=True, type=str, help="Path to parameter json file")
    parser.add_argument('-m','--mode', required=True, type=str, help="Pipeline step to run. Choose from `model_training`, `feature_extraction`, `feature_analysis` or `few_shot`")
    args = parser.parse_args()
    main(args.params_filename, args.mode)
