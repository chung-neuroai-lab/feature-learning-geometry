import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pipeline_dataset
import pipeline_utils
import run_feature_analysis
from copy import deepcopy
import gcmc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CIFAR10C(Dataset):
    def __init__(self, data_folder, corrupt_type, severity, transform=None):
        self.root_dir = data_folder
        self.transfrom = transform
        self.severity = severity
        self.start_idx, self.end_idx = self._severity_to_idx()
        self.data_path = os.path.join(self.root_dir, f"{corrupt_type}.npy") # test first with `gaussian_noise`
        self.target_path = os.path.join(self.root_dir, "labels.npy")
        self.data = np.load(self.data_path)[self.start_idx : self.end_idx] # first 10000 images are with severity = 1, and so on...
        self.targets = list(int(i) for i in np.load(self.target_path))[self.start_idx : self.end_idx] # convert to `int` so `DataLoader` has torch.int64
        self.classes = list(set(self.targets))
    
    def _severity_to_idx(self): # map severity level to idx in dataset
        severity_to_idx = {i+1: (i*10000, (i+1)*10000) for i in range(5)} # eg {1: [0, 10000]}
        return severity_to_idx[self.severity]
    
    def __getitem__(self, idx):
        
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img) # return PIL image to be consistent with all other datasets
        if self.transfrom is not None:
            img = self.transfrom(img)
        return img, target, idx
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"CIFAR10C dataset at path {self.data_path} with severity {self.severity}"
    
@torch.no_grad
def run_test(model, model_0, criterion, testloader, params):
    test_loss = 0
    test_loss_scaled = 0
    correct = 0
    total = 0
    alpha = params["alpha"]

    for batch_idx, (inputs, targets, idx) in enumerate(testloader):
        inputs, targets = inputs.to(DEVICE).float(), targets.to(DEVICE)
        if params["precision"] == "double":
            inputs=inputs.double()
        outputs_ = model(inputs)
        outputs_0 = model_0(inputs)
        outputs = outputs_ - outputs_0

        targets_ = targets.unsqueeze(1)
        if DEVICE == "cuda":
            targets_embed = torch.zeros(targets_.size(0), 10).cuda()
        else:
            targets_embed = torch.zeros(targets_.size(0), 10)
        targets_embed.scatter_(1, targets_, 1)
        loss = criterion(outputs, targets_embed)
        loss_scaled = criterion(outputs, targets_embed / alpha)

        test_loss_scaled += loss_scaled.item()
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss/(batch_idx+1)
    test_loss_scaled = test_loss_scaled / (batch_idx+1)
    test_acc = 100.*correct/total
    print(f'Log Loss: {np.log(test_loss)} | Log Loss scaled: {np.log(test_loss_scaled)} | Acc: {test_acc: .3f}%% ({correct}/{total})')    
    return test_loss, test_acc, test_loss_scaled

def save_eval_result(state_list, filename_config, filename_params):
    import pickle
    filepath = filename_config["eval_corrupt"].format(**filename_params)
    parent_folder = os.path.dirname(filepath)
    if not os.path.isdir(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)
    # If already exist a past state, append to that past state
    if os.path.exists(filepath):
        past_state = pickle.load(open(filepath, "rb"))
        state_list = past_state + state_list
    with open(filepath, 'wb') as f:
        pickle.dump(state_list, f)
    print(f"Saving eval corrupt results to file path {filepath}")

def main(params):
    print(f"=== Process: Evaluate on corrupted dataset... ===")
    print(f"Parameters: {params}")
    if params["precision"] == "float":
        torch.set_default_dtype(torch.float32)
    elif params["precision"] == "double":
        torch.set_default_dtype(torch.float64)

    ### Extract parameters
    save_folder = params["save_folder"]
    corrupt_mode = params["mode"]
    training_mode = "model_training"
    filename_config = pipeline_utils.get_config()["filename_config"]
    corrupt_data_folder = filename_config["corrupt_data_folder"]
    model_name = params["model"]
    num_epoch = params["num_epoch"]
    epoch_list = pipeline_utils.generate_epoch_list(num_epoch)
    criterion = pipeline_utils.get_criterion(params["criterion"])
    run_capacity = bool(params["runCapacity"]) if "runCapacity" in params else False
    run_eval = bool(params["runEvalCorrupt"]) if "runEvalCorrupt" in params else True
    sample_function_name = params["sampleFunc"]
    feature_sample_size = params["numSmpl"]
    num_class_selected = params["numCls"]
    sample_function = pipeline_dataset.get_sample_function_by_name(sample_function_name)
    layer = params["layer"]
    sub_seed = params["subSeed"] if "subSeed" in params else 0
    severity_list = params["corruptLevels"]

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
        
        # Load corrupted dataset
        corrupt_types = params["fsData"]
        input_folder = filename_config["input_data"].format(save_folder=params["save_folder"])
        train_set = pipeline_dataset.CIFAR10WithIndices(
            root=input_folder, train=True, transform=transforms.Compose([transforms.ToTensor()]))
        normalization_params = pipeline_dataset.CIFAR10Dataset.get_normalization_params(train_set)
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalization_params["mean"], normalization_params["std"]),
        ])
        
        for corrupt_type in corrupt_types:
            eval_results = []
            filename_base = pipeline_utils.generate_filename({**params, "fsData": corrupt_type}, corrupt_mode)
            for severity in severity_list:
                print(f"Corrupt type: {corrupt_type}, severity level {severity}")
                corrupt_dataset = CIFAR10C(data_folder=corrupt_data_folder, corrupt_type=corrupt_type, severity=severity, transform=transform_test)
                corrupt_dataloader = DataLoader(corrupt_dataset, batch_size=100, shuffle=False)
                label_list = list(set(corrupt_dataset.targets))
                if num_class_selected < len(label_list):
                    selected_cls = np.random.permutation(label_list)[:num_class_selected]
                else:
                    selected_cls = label_list
                model_filepath_0 = model_checkpoint_pattern.format(**model_checkpoint_filename_params,
                                                epoch=0)
                base_model_0 = pipeline_utils.load_base_model(model_name)
                model_0 = pipeline_utils.load_model_from_path(base_model_0, model_filepath_0)
                model_0 = model_0.to(DEVICE); model_0.eval()

                for epoch in epoch_list:
                    model_filepath = model_checkpoint_pattern.format(**model_checkpoint_filename_params,
                                                epoch=epoch)
                    base_model = pipeline_utils.load_base_model(model_name)
                    model = pipeline_utils.load_model_from_path(base_model, model_filepath)
                    model = model.to(DEVICE); model.eval()
                    if run_eval:
                        test_loss, test_acc, test_loss_scaled = run_test(model, model_0, criterion, corrupt_dataloader, params)
                        eval_results.append({
                            "epoch": epoch,
                            "severity": severity,
                            "test_loss": test_loss,
                            "test_acc": test_acc,
                            "test_loss_scaled": test_loss_scaled,
                        })
                    # Measure capacity of the corrupted dataset
                    if run_capacity:
                        # Get sample dataset
                        sample_function_params = {
                        "dataset": corrupt_dataset,
                        "num_sample": feature_sample_size,
                        "model": model,
                        "criterion": criterion,
                        "selected_labels": selected_cls,
                        "seed": sub_seed,
                        }
                        sample_data_idx = sample_function(**sample_function_params)
                        sample_dataset = torch.utils.data.Subset(corrupt_dataset, sample_data_idx)
                        sample_dataloader = torch.utils.data.DataLoader(
                            sample_dataset, batch_size=100, shuffle=False)
                        capacity_filename_base = pipeline_utils.generate_filename({**params, "fsData": corrupt_type}, mode="feature_analysis")
                        # Extract features and run capacity
                        analysis_results = {}
                        features = run_feature_analysis.extract_features(model, sample_dataloader, [layer])[layer]
                        feature_data = run_feature_analysis.prepare_data_for_manifold_analysis(features)
                        print(f"Num classes: {len(feature_data)} | Shape: {feature_data[0].shape}")
                        print("Running manifold analysis...")
                        try:
                            analysis_result = gcmc.manifold_analysis(feature_data, backend="numpy")
                            # analysis_result = pipeline_utils.compute_capacity(feature_data)
                            analysis_results[layer] = analysis_result
                        except:
                            try:
                                feature_data += np.random.randn(*feature_data.shape)*1e-6
                                analysis_result = gcmc.manifold_analysis(feature_data, backend="numpy")
                                # analysis_result = pipeline_utils.compute_capacity(feature_data)
                                analysis_results[layer] = analysis_result
                            except Exception as e:
                                print(f"Run to error with params {params} at epoch {epoch} layer {layer} \n error msg {e}")
                        # Save analysis results
                        result_filename_params =   {"save_folder": save_folder,
                                        "mode": corrupt_mode,
                                        "filename_base": capacity_filename_base,
                                        "epoch": epoch,
                                        "severity": severity,
                                        }
                        pipeline_utils.save_analysis_result(analysis_results, filename_config, result_filename_params, save_mode="eval_corrupt_capacity")
            #Save to file
            if run_eval:
                eval_filename_params = {"save_folder": save_folder,
                                        "mode": corrupt_mode,
                                        "filename_base": filename_base}
                save_eval_result(eval_results, filename_config, eval_filename_params)
