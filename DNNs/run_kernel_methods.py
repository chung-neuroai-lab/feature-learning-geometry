import pipeline_dataset
import pipeline_utils
import os
import torch
import numpy as np
from copy import deepcopy
from torch.func import functional_call, vmap, jacrev

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Could not use `vmap` due to out-of-memory error
# def compute_jacobian(fnet_single, params, x):
#     jac_res = vmap(jacrev(fnet_single), (None, 0))(params, x)
#     jac = []
#     for j in jac_res.values():
#         j_size = j.size()
#         batch_size, num_class = j_size[0], j_size[1]
#         jac.append(j.view(batch_size*num_class, -1).cpu())
#     del jac_res # release cuda mem
#     jac = torch.cat(jac, dim=1)
#     return jac
#
# class NetWrapper(torch.nn.Module):
#     def __init__(self, init_model, trained_model):
#         super(NetWrapper, self).__init__()
#         self.init_model = init_model
#         self.trained_model = trained_model

#     def forward(self, x):
#         x = self.trained_model(x) - self.init_model(x)
#         return x

def compute_jacobian(inputs, model, model_0, num_classes):
    net_parameters = list(model.parameters())
    params = sum([torch.numel(p) for p in net_parameters]) # number of parameters
    output_linearized = torch.zeros(inputs.size(0), num_classes, params) # the neural tangent
    output_t = model(inputs)
    output_0 = model_0(inputs)
    output = output_t - output_0 # specific implementation to keep initial outputs at zero for lazy training. see Chizat et al 2018.
    # iterate each sample, class, net parameters to avoid OOM error
    for n in range(inputs.size(0)): # iterate each sample
        for i in range(num_classes): # iterate each class
            output[n, i].backward(retain_graph=True) # do backprop
            p_idx = 0
            for p in range(len(net_parameters)): # iterate each parameter
                this_grad = net_parameters[p].grad.data.view(-1)
                output_linearized[n, i, p_idx: p_idx+net_parameters[p].numel()] = this_grad
                p_idx = p_idx + net_parameters[p].numel()
            for p in range(len(net_parameters)):
                net_parameters[p].grad.data.zero_()
    output_linearized = output_linearized.view(inputs.size(0)*num_classes,params)
    return output_linearized

def extract_ntk(data_loader, data_loader2, model, model_0, num_classes):
    
    num_sample = len(data_loader.dataset)
    ntk_length = num_sample * num_classes
    ntk = torch.zeros([ntk_length, ntk_length])
    idx, idx2 = 0, 0

    for batch_idx, (inputs, targets, _) in enumerate(data_loader):
        print(f"Batch idx: {batch_idx}")
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        out = compute_jacobian(inputs, model, model_0, num_classes)

        for batch_idx2, (inputs2, targets2, _) in enumerate(data_loader2):
            if batch_idx2 < batch_idx:
                idx2 = idx2 + inputs2.size(0)*num_classes #K is symmetric, so we only need to compute the upper half
                continue
            inputs2, targets2 = inputs2.to(DEVICE), targets2.to(DEVICE)
            out2 = compute_jacobian(inputs, model, model_0, num_classes)
            ntk_sub = torch.mm(out, out2.t())
            ntk[idx:idx+num_classes*inputs.size(0), idx2:idx2+num_classes*inputs2.size(0)] = ntk_sub
            ntk[idx2:idx2 + num_classes*inputs2.size(0), idx:idx +num_classes*inputs.size(0)] = ntk_sub.t() #K is symmetric, so we only need to compute the upper half
            idx2 = idx2 + num_classes*inputs2.size(0)
            del out2 # release memory
        idx2 = 0
        idx = idx + inputs.size(0)*num_classes
        del out # release memory

    return ntk.numpy()

def compute_cka(ntk, y, num_classes, num_sample_per_class):
    cka = np.zeros((num_classes))
    num_sample = num_classes * num_sample_per_class
    y = y.reshape((-1, 1)) # shape (num_sample, 1)
    y_onehot = np.zeros((len(y), num_classes))
    np.put_along_axis(y_onehot, y, 1, axis=1) # convert into one-hot labels
    assert np.all(np.sum(y_onehot, axis=1)) == 1, "One-hot encoding must have 1 positive class per sample!"
    y_onehot = y_onehot.T # shape (num_class, num_sample)

    for class_idx in range(num_classes):
        sub_ntk_idx = [i*num_classes+class_idx for i in range(num_sample)]
        sub_ntk = np.array([ntk[i, sub_ntk_idx] for i in sub_ntk_idx]) # shape (num_sample, num_sample)
        sub_onehot = y_onehot[class_idx].reshape((-1, 1)) # shape (num_sample, 1)
        sub_cka = (sub_onehot.T @ sub_ntk @ sub_onehot) / (np.linalg.norm(sub_onehot)**2 * np.linalg.norm(sub_ntk))
        cka[class_idx] = sub_cka.item()
    mean_cka = np.mean(cka)
    return mean_cka

def compute_ntk_change(ntk_0, ntk_1):
    ntk_change = np.linalg.norm(ntk_1 - ntk_0) / np.linalg.norm(ntk_0)
    return ntk_change

def main(params):

    print(f"=== Process: Computing kernels... ===")
    print(f"Parameters: {params}")

    ### Extract parameters
    save_folder = params["save_folder"]
    kernel_mode = params["mode"]
    training_mode = "model_training"
    filename_config = pipeline_utils.get_config()["filename_config"]
    model_name = params["model"]
    num_epoch = params["num_epoch"]
    sub_seed = params["subSeed"] if "subSeed" in params else 0
    np.random.seed(sub_seed)
    use_step = bool(params["useStep"])
    epoch_list = pipeline_utils.generate_epoch_list(num_epoch, use_step=use_step)
    use_last_epoch = params["lastEpoch"] if "lastEpoch" in params else False
    if use_last_epoch:
        epoch_list = [epoch_list[-1]]

    if isinstance(params["alpha"], list):
        alphas = deepcopy(params["alpha"])
    else:
        alphas = [deepcopy(params["alpha"])]
    for alpha in alphas:
        print(f"Processing alpha {alpha}...")
        params["alpha"] = alpha
        # Model checkpoint filename
        model_checkpoint_pattern = filename_config["model_checkpoint"]
        model_checkpoint_filename_base = pipeline_utils.generate_filename(params, training_mode)
        model_checkpoint_filename_params = {"save_folder": save_folder,
                                            "filename_base": model_checkpoint_filename_base,
                                            "mode": training_mode
                                            }
        dataset_name = params["fsData"]
        input_folder = filename_config["input_data"]
        sample_function_name = params["sampleFunc"]
        feature_sample_size = params["numSmpl"]
        num_classes = params["numCls"]

        # Get the whole dataset
        dataset_obj = pipeline_dataset.get_dataset_obj_by_name(dataset_name, input_folder)
        dataset_dict = dataset_obj.get_dataset()
        test_set = dataset_dict["test_set"]
        sample_function = pipeline_dataset.get_sample_function_by_name(sample_function_name)
        sample_data_idx = sample_function(test_set, feature_sample_size, None, None, seed=sub_seed)
        sample_dataset = torch.utils.data.Subset(test_set, sample_data_idx)
        sample_data_loader = torch.utils.data.DataLoader(
            sample_dataset, batch_size=50, shuffle=False)
        sample_data_loader2 = torch.utils.data.DataLoader(
            sample_dataset, batch_size=50, shuffle=False)
        sample_targets = np.array(sample_dataset.dataset.targets)[sample_data_idx]

        kernel_filename_base = pipeline_utils.generate_filename(params, kernel_mode)
        model_filepath_0 = model_checkpoint_pattern.format(**model_checkpoint_filename_params,
                                        epoch=0)
        model_0 = pipeline_utils.load_model_from_path(pipeline_utils.load_base_model(model_name), model_filepath_0)
        model_0 = model_0.to(DEVICE)
        ntk_0 = None

        result_list = []
        for epoch_idx in range(len(epoch_list)):
            epoch = epoch_list[epoch_idx]
            print(f"Extracting epoch {epoch}. [{epoch_idx} / {len(epoch_list)}] epoch...")
            model_filepath = model_checkpoint_pattern.format(**model_checkpoint_filename_params,
                                            epoch=epoch)
            model = pipeline_utils.load_model_from_path(pipeline_utils.load_base_model(model_name), model_filepath)
            model = model.to(DEVICE)
            ntk = extract_ntk(sample_data_loader, sample_data_loader2, model, model_0, num_classes)
            if epoch_idx == 0:
                ntk_0 = deepcopy(ntk)
            cka = compute_cka(ntk, sample_targets, num_classes, feature_sample_size)
            ntk_change = compute_ntk_change(ntk_0, ntk)
            res = {
                "epoch": epoch,
                "ntk": ntk,
                "ntk_change": ntk_change,
                "cka": cka,
            }
            result_list.append(res)
            del model # release memory
        # Save analysis results
        result_filename_params =   {"save_folder": save_folder,
                        "mode": kernel_mode,
                        "filename_base": kernel_filename_base,
                        }
        pipeline_utils.save_state_list(result_list, filename_config, result_filename_params, kernel_mode)
