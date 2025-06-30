def validate_params(input_params, valid_configs):
    for param_name, param_vals in input_params.items():
        if param_name in valid_configs:
            valid_params = set(valid_configs[param_name])
            for param_val in param_vals:
                if isinstance(param_val, list):
                    if not set(param_val).issubset(valid_params):
                        raise ValueError(f"Param {param_val} is not a valid parameter for param {param_name}. Valid params are {valid_params}!")
                else:
                    if param_val not in valid_params:
                        raise ValueError(f"Param {param_val} is not a valid parameter for param {param_name}. Valid params are {valid_params}!")
    return True

def generate_parameter_list(inputs, intermediate_results=[]):
    """Generate list of parameters with inputs as dict with keys as parameter names,
    values are list of possible parameter values, and output as list of all possible dict
    with keys are parameter names, values are parameter values.

    Args:
        inputs (dict<str, list>): Dict with keys as parameter names, values are list of possible parameter values
        intermediate_results (list, optional): Intermediate results of parameter list. Defaults to [].

    Returns:
        list(dict<str, any>): List of all possible dict with keys are parameter anems, values are parameter values.
    """
    if len(inputs) == 0: #base case
        return intermediate_results
    if len(inputs) > 0: # recursive case
        new_intermediate_results = []
        key, vals = inputs.popitem()
        for val in vals:
            if len(intermediate_results) == 0: # edge case when intermediate result is empty
                new_intermediate_results.append({key: val})
            else:
                for res in intermediate_results:
                    new_intermediate_results.append({**res, key: val})
        return generate_parameter_list(inputs, intermediate_results=new_intermediate_results)

def generate_parameter_pipeline(param_inputs, pipeline_name):
    # TODO: Move this to `config.json`
    if pipeline_name == "standard":
        pipeline_steps = ["model_training", "feature_analysis"]
    elif pipeline_name == "linear_probe":
        pipeline_steps = ["model_training", "linear_probe"]
    elif pipeline_name == "eval_corrupt":
        pipeline_steps = ["model_training", "eval_corrupt"]
    elif pipeline_name == "weight_analysis":
        pipeline_steps = ["model_training", "weight_analysis"]
    elif pipeline_name == "kernels":
        pipeline_steps = ["model_training", "kernels"]
    else:
        raise ValueError(f"Pipeline {pipeline_name} is not supported!")
    params = {}
    for i in range(len(pipeline_steps)):
        this_step = pipeline_steps[i]
        this_step_params = {}
        for j in range(i+1):
            prev_step = pipeline_steps[j]
            this_step_params = {**this_step_params, **param_inputs[prev_step]}
        params[this_step] = generate_parameter_list(this_step_params)
    return params

def generate_filename(params, mode):
    # TODO: Move this to `config.json`
    mode_to_params = {
        "model_training": sorted(["model", "task", "optim", "lr", "alpha", "decay", "seed"]),
        "feature_extraction": sorted(["model", "task", "optim", "lr", "alpha", "sampleFunc", "fsData", "seed", "decay", "numSmpl", "numCls", "subSeed"]),
        "feature_analysis": sorted(["model", "task", "optim", "lr", "alpha", "sampleFunc", "fsData", "seed", "decay", "numSmpl", "numCls", "subSeed"]),
        "few_shot": sorted(["model", "task", "optim", "lr", "alpha", "fsWay", "fsShot", "seed", "decay"]),
        "linear_probe": sorted(["model", "task", "optim", "lr", "alpha", "fsData", "decay", "numCls", "seed"]),
        "eval_corrupt": sorted(["model", "task", "optim", "lr", "alpha", "fsData", "decay", "seed"]),
        "weight_analysis": sorted(["model", "task", "optim", "lr", "alpha", "decay", "seed"]),
        "kernels": sorted(["model", "task", "optim", "lr", "alpha", "decay", "seed", "numSpl", "numCls", "subSeed"]),
    }
    if mode not in mode_to_params:
        raise ValueError(f"Mode {mode} is not in supported list of mode {mode_to_params.keys()}")
    else:
        filename = ""
        included_params = mode_to_params[mode]
        for i in range(len(included_params)):
            param = included_params[i]
            if param in params:
                if i != len(included_params) - 1:
                    filename += f"{param}-{params[param]}_"
                else:
                    filename += f"{param}-{params[param]}"
        return filename

def get_steps(num_step):
    """Given number of gradient steps (`num_step` int), return the list of chosen steps (`chosen_step` [int])
    to be used to extract features from, typically double each step ([0, 1, 2, 4, 8, ...]). Also include
    mid-point for long-step range (step, int(step*4/3), int(step*5/3), step*2)
    """
    double_steps = [0, 1]
    current_step = 1
    while current_step*2 < num_step:
        current_step = current_step*2
        double_steps.append(current_step)
    chosen_steps = []
    for s in range(len(double_steps)-1):
        step = double_steps[s]
        chosen_steps.append(step)
        if step >= 4: # also add mid-point for long-step range
            chosen_steps.append(int(step*4/3))
            chosen_steps.append(int(step*5/3))
    chosen_steps.append(double_steps[-1])
    return chosen_steps

def generate_epoch_list(num_epoch, use_step=False):
    # TODO: Generalize this function
    if not use_step:
        base_epoch = [i for i in range(10)] + [10+2*i for i in range(10)] + [30+5*i for i in range(10)] + [80+10*i for i in range(192)] + [2000+20*i for i in range(100)]
        epoch_list = [i for i in base_epoch if i < num_epoch]
    else:
        epoch_list = get_steps(num_epoch)
    return epoch_list

def get_config():
    import json
    config_filename = "config.json"
    config = json.load(open(config_filename, "rb"))
    return config

def load_model_from_path(model, model_filepath):
    import torch
    with torch.no_grad():
        with open(model_filepath, 'rb') as f:
            state = torch.load(f, map_location=torch.device('cpu'))
        state_dict = state['model']
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            if "Missing key(s) in state_dict" in str(e):
                # replace .module due to nn.DataParallel wraper class
                model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
            else:
                raise e
    return model

def load_base_model(model_name, pretrained=False, k=1):
    # TODO: Use `models_cifar` for all resnet on cifar
    if model_name == "resnet18":
        import models_lazy
        model = models_lazy.ResNet18(k)
    elif "VGG" in model_name:
        from models_lazy import VGG
        model = VGG(model_name, k)
    else:
        import torchvision
        model = getattr(torchvision.models, model_name)(pretrained=pretrained)
    return model

def get_criterion(criterion_name, reduction="mean"):
    import torch
    if criterion_name == "ce":
        criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    elif criterion_name == "mse":
        criterion = torch.nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"Criterion {criterion_name} is not in supported criterion list [ce, mse]")
    return criterion

def save_state_list(state_list, filename_config, filename_params, save_mode):
    import os
    import pickle
    
    filepath = filename_config[save_mode].format(**filename_params)
    parent_folder = os.path.dirname(filepath)
    if not os.path.isdir(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)
    # If already exist a past state, append to that past state
    if os.path.exists(filepath):
        past_state = pickle.load(open(filepath, "rb"))
        state_list = past_state + state_list
    with open(filepath, 'wb') as f:
        pickle.dump(state_list, f)
    print(f"Saving mode {save_mode} file path {filepath}")

def save_analysis_result(results, filename_config, filename_params, save_mode):
    import pickle
    import os

    filepath = filename_config[save_mode].format(**filename_params)
    parent_folder = os.path.dirname(filepath)
    if not os.path.isdir(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saving mode {save_mode} to {filepath}...")

def compute_capacity_pairwise(XtotTs):
    """ Use pairwise computation for capacity in case of highly correlated data
    """
    import pandas as pd
    from gcmc.contrib import gcmc_analysis_dataframe
    df_list = []
    num_cls, num_dim, num_sample = XtotTs.shape
    for i in range(num_cls):
        for j in range(i+1, num_cls):
            XtotT = [XtotTs[i], XtotTs[j]]
            ret = gcmc_analysis_dataframe(XtotT,indices=(i,j),indices_name=['i','j'])
            df_list.append(ret)
    df = pd.concat(df_list)
    return df
