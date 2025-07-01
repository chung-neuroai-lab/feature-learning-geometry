import torch
import torch.nn as nn
import os
import pickle
from torchvision.models.feature_extraction import create_feature_extractor
import pipeline_dataset
import pipeline_utils
from copy import deepcopy
import numpy as np
import run_feature_analysis
import gcmc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EarlyStopper:
    def __init__(self, patience=3, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class LinearProbeModel(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(LinearProbeModel, self).__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
class LinearProbe:
    def __init__(self, pretrained_model, in_dim, num_classes, layer="view"):
        self.pretrained_model = pretrained_model; pretrained_model.eval(); pretrained_model.to(DEVICE)
        self.layer = layer # the layer of the pre-trained model to extract feature from
        self.feature_extractor = create_feature_extractor(self.pretrained_model, return_nodes=[self.layer])
        self.linear_probe = LinearProbeModel(in_dim, num_classes); self.linear_probe.to(DEVICE)

    def run_train(self, dataloader, criterion, optimizer):
        self.linear_probe.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, idx) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            features = self.feature_extractor(inputs)[self.layer]
            outputs = self.linear_probe(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = train_loss/(batch_idx+1)
        train_acc = 100.*correct/total
        print(f'[Train] Loss: {train_loss: .3f} | Acc: {train_acc: .3f}%% ({correct}/{total})')   
        return train_loss, train_acc

    @torch.no_grad
    def run_test(self, dataloader, criterion):
        self.linear_probe.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, idx) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            features = self.feature_extractor(inputs)[self.layer]
            outputs = self.linear_probe(features)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_loss = test_loss/(batch_idx+1)
        test_acc = 100.*correct/total
        print(f'[Test] Loss: {test_loss: .3f} | Acc: {test_acc: .3f}%% ({correct}/{total})')    
        return test_loss, test_acc
    
    @torch.no_grad
    def run_predict(self, inputs):
        self.linear_probe.eval()
        inputs = inputs.to(DEVICE)
        features = self.feature_extractor(inputs)[self.layer]
        outputs = self.linear_probe(features)
        _, prediction = outputs.max(1)
        return prediction

def save_checkpoint(model, filename_config, filename_params):

    state = {'model': model.state_dict()}
    filepath = filename_config["linear_probe_checkpoint"].format(**filename_params)
    parent_folder = os.path.dirname(filepath)
    if not os.path.isdir(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)
    torch.save(state, filepath)
    print(f'Saving epoch {filename_params["epoch"]} to file path {filepath}...')

def save_training_state(state_list, filename_config, filename_params):
    filepath = filename_config["linear_probe_state"].format(**filename_params)
    parent_folder = os.path.dirname(filepath)
    if not os.path.isdir(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)
    # If already exist a past state, append to that past state
    if os.path.exists(filepath):
        past_state = pickle.load(open(filepath, "rb"))
        state_list = past_state + state_list
    with open(filepath, 'wb') as f:
        pickle.dump(state_list, f)
    print(f"Saving training states (acc, loss) to file path {filepath}")

def get_selected_cls_sample(selected_cls, targets):
    sample_idx = []
    selected_cls_set = set(selected_cls)
    for i in range(len(targets)):
        if targets[i] in selected_cls_set:
            sample_idx.append(i)
    return sample_idx
    
def main(params):

    print(f"=== Process: Run Linear Probe... ===")
    print(f"Parameters: {params}")

    ### Extract parameters
    save_folder = params["save_folder"]
    linear_probe_mode = params["mode"]
    training_mode = "model_training"
    filename_config = pipeline_utils.get_config()["filename_config"]
    model_name = params["model"]
    num_epoch = params["num_epoch"]
    sub_seed = params["subSeed"] if "subSeed" in params else 0
    np.random.seed(sub_seed)
    layer = params["layer"]
    run_capacity = bool(params["runCapacity"]) if "runCapacity" in params else False
    run_probe = bool(params["runLinearProbe"]) if "runLinearProbe" in params else True
    epoch_list = pipeline_utils.generate_epoch_list(num_epoch)
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

        ### Model checkpoint filename
        model_checkpoint_pattern = filename_config["model_checkpoint"]
        model_checkpoint_filename_base = pipeline_utils.generate_filename(params, training_mode)
        model_checkpoint_filename_params = {"save_folder": save_folder,
                                            "filename_base": model_checkpoint_filename_base,
                                            "mode": training_mode
                                            }

        ### Load linear probe dataset
        lp_dataset_name = params["fsData"]
        input_folder = filename_config["input_data"]
        sample_function_name = params["sampleFunc"]
        sample_function = pipeline_dataset.get_sample_function_by_name(sample_function_name)
        feature_sample_size = params["numSmpl"]
        num_class_selected = params["numCls"]
        dataset_dict = pipeline_dataset.get_dataset_obj_by_name(lp_dataset_name, input_folder).get_dataset()
        train_set, train_loader, test_set, test_loader = dataset_dict["train_set"], dataset_dict["train_loader"], dataset_dict["test_set"], dataset_dict["test_loader"]

        # Create subset dataset
        base_model = pipeline_utils.load_base_model(model_name)
        model_filepath = model_checkpoint_pattern.format(**model_checkpoint_filename_params,
                                            epoch=epoch_list[-1])
        model_last_epoch = pipeline_utils.load_model_from_path(base_model, model_filepath)
        label_list = list(set(test_set.targets))
        if num_class_selected < len(label_list):
            selected_cls = np.random.permutation(label_list)[:num_class_selected]
        else:
            selected_cls = label_list

        ### Linear probe filename and checkpoint
        num_epoch_lp = 50
        in_dim = 512 # TODO: replace this hard-code value
        lr = 0.1
        num_classes = len(train_set.classes)

        filename_config = pipeline_utils.get_config()["filename_config"]
        filename_base = pipeline_utils.generate_filename(params, params["mode"])
        filename_params_base = {"save_folder": save_folder, "filename_base": filename_base, "mode": params["mode"]}

        ### Train linear probe
        for (epoch_idx, epoch) in enumerate(epoch_list):
            print(f"Extracting epoch {epoch}. [{epoch_idx} / {len(epoch_list)}] epoch...")
            # Load pre-trained model
            model_filepath = model_checkpoint_pattern.format(**model_checkpoint_filename_params,
                                    epoch=epoch)
            base_model = pipeline_utils.load_base_model(model_name)
            model = pipeline_utils.load_model_from_path(base_model, model_filepath)
            print(f"Loading model at {model_filepath}")
            model = model.to(DEVICE)
            model.eval()
            if run_probe:
                linear_probe = LinearProbe(model, in_dim, num_classes)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(linear_probe.linear_probe.parameters(), lr=lr)
                early_stopper = EarlyStopper()
                checkpoint_filename_params = {**filename_params_base, "epoch": epoch}
                best_test_acc = 0.0
                result_list_lp = []
                # Train linear probe
                for epoch_lp in range(num_epoch_lp):
                    print(f"Training epoch {epoch_lp} / {num_epoch_lp}...")
                    if epoch_lp == 0:
                        print("Untrained model")
                        train_loss, train_acc = linear_probe.run_test(train_loader, criterion)
                        test_loss, test_acc = linear_probe.run_test(test_loader, criterion)
                    else:
                        this_lr = lr /(1.0+ (epoch_lp-1)/3)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = this_lr
                        train_loss, train_acc = linear_probe.run_train(train_loader, criterion, optimizer)
                        test_loss, test_acc = linear_probe.run_test(test_loader, criterion)
                        if early_stopper.early_stop(test_loss):
                            break
                    
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        save_checkpoint(linear_probe.linear_probe, filename_config, checkpoint_filename_params)
                    
                    result_list_lp.append({
                        "epoch_lp": epoch_lp,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                    })
                save_training_state(result_list_lp, filename_config, checkpoint_filename_params)
            
            if run_capacity:
                # Get sample dataset
                sample_function_params = {
                    "dataset": test_set,
                    "num_sample": feature_sample_size,
                    "model": model_last_epoch,
                    "criterion": criterion,
                    "selected_labels": selected_cls,
                    "seed": sub_seed,
                }
                capacity_data_idx = sample_function(**sample_function_params)
                capacity_dataset = torch.utils.data.Subset(test_set, capacity_data_idx)
                capacity_dataloader = torch.utils.data.DataLoader(
                    capacity_dataset, batch_size=100, shuffle=False)
                capacity_filename_base = pipeline_utils.generate_filename(params, mode="feature_analysis")
                # Extract features and run capacity
                analysis_results = {}
                features = run_feature_analysis.extract_features(model, capacity_dataloader, [layer])[layer]
                feature_data = run_feature_analysis.prepare_data_for_manifold_analysis(features)
                print(f"Num classes: {len(feature_data)} | Shape: {feature_data[0].shape}")
                print("Running manifold analysis...")
                try:
                    analysis_result = pipeline_utils.compute_capacity_pairwise(feature_data)
                    analysis_results[layer] = analysis_result
                except:
                    try:
                        feature_data += np.random.randn(*feature_data.shape)*1e-6 # add noise to avoid rank error
                        analysis_result = pipeline_utils.compute_capacity_pairwise(feature_data)
                        analysis_results[layer] = analysis_result
                    except Exception as e:
                        print(f"Run to error with params {params} at epoch {epoch} layer {layer} \n error msg {e}")
                # Save analysis results
                result_filename_params =   {"save_folder": save_folder,
                                "mode": linear_probe_mode,
                                "filename_base": capacity_filename_base,
                                "epoch": epoch,
                                }
                pipeline_utils.save_analysis_result(analysis_results, filename_config, result_filename_params, save_mode="linear_probe_capacity")
