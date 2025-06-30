import pipeline_dataset
import pipeline_utils
import os
import torch
import numpy as np
from copy import deepcopy
import gcmc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad
def extract_features(model, data_loader, layers):
    from torchvision.models.feature_extraction import create_feature_extractor
    extractor = create_feature_extractor(model, return_nodes=layers)
    features = {}
    for (input, target, idx) in data_loader:
        input = input.to(DEVICE)
        feature = extractor(input)
        target = target.detach().to("cpu").numpy()
        idx = idx.detach().to("cpu").numpy()
        for key, val in feature.items():
            val = val.detach().to("cpu").numpy()
            if key in features:
                features[key]["feature"].append(val)
                features[key]["label"].append(target)
                features[key]["idx"].append(idx)
            else:
                features[key] = {}
                features[key]["feature"] = [val]
                features[key]["label"] = [target]
                features[key]["idx"] = [idx]

    for layer in features:
        for key in features[layer]:    
            if key == "feature":
                features[layer][key] = np.vstack([i.reshape(i.shape[0], -1) for i in features[layer][key]])
                features[layer][key] = random_projection(features[layer][key])
            else: # `label` and `idx` key
                features[layer][key] = np.concatenate(features[layer][key])
    return features

def random_projection(X, dim=1000):
    import numpy as np

    N = X.shape[1]
    if N > dim:
        print(f"Projecting from {N} dim to {dim} dim...")
        M = np.random.randn(dim, N)
        M /= np.sqrt(np.sum(M*M, axis=1, keepdims=True))
        X = np.vstack([np.matmul(M, d) for d in X])
    return X

def save_features(features, filename_config, filename_params):
    import pickle
    filepath = filename_config["extracted_feature"].format(**filename_params)
    parent_folder = os.path.dirname(filepath)
    if not os.path.isdir(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(features, f)
    print(f"Saving extracted features to {filepath}...")

def prepare_data_for_manifold_analysis(input_dict):
    unique_labels = np.unique(input_dict["label"])
    out = []
    for label in unique_labels:
        out.append(input_dict["feature"][input_dict["label"] == label, :].T)
    out = np.stack(out)
    return out

def main(params):

    print(f"=== Process: Extracting features ===")
    print(f"Parameters: {params}")

    ### Extract parameters
    save_folder = params["save_folder"]
    analysis_mode = params["mode"]
    extraction_mode = "feature_extraction"
    training_mode = "model_training"
    filename_config = pipeline_utils.get_config()["filename_config"]
    model_name = params["model"]
    num_epoch = params["num_epoch"]
    layers = params["layers"]
    sub_seed = params["subSeed"] if "subSeed" in params else 0
    np.random.seed(sub_seed)
    use_step = params["useStep"]
    epoch_list = pipeline_utils.generate_epoch_list(num_epoch, use_step=use_step)
    use_last_epoch = params["lastEpoch"] if "lastEpoch" in params else False
    do_save_feature = params["saveFeature"] if "saveFeature" in params else False
    do_run_analysis = params["runAnalysis"] if "runAnalysis" in params else True
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

        ### Load dataset (usually sample subset of the whole dataset)
        dataset_name = params["fsData"]
        input_folder = filename_config["input_data"]
        sample_function_name = params["sampleFunc"]
        feature_sample_size = params["numSmpl"]
        num_class_selected = params["numCls"]

        # Get the whole dataset
        dataset_obj = pipeline_dataset.get_dataset_obj_by_name(dataset_name, input_folder)
        dataset_dict = dataset_obj.get_dataset()
        test_set = dataset_dict["test_set"]
        sample_function = pipeline_dataset.get_sample_function_by_name(sample_function_name)
        criterion = pipeline_utils.get_criterion(params["criterion"], reduction="none")

        # Feature filename
        analysis_filename_base = pipeline_utils.generate_filename(params, analysis_mode)
        feature_filename_base = pipeline_utils.generate_filename(params, extraction_mode)

        # Create the subset dataset from last epoch model
        base_model = pipeline_utils.load_base_model(model_name)
        model_filepath = model_checkpoint_pattern.format(**model_checkpoint_filename_params,
                                            epoch=epoch_list[-1])
        model_last_epoch = pipeline_utils.load_model_from_path(base_model, model_filepath)
        label_list = list(set(test_set.targets))
        if num_class_selected < len(label_list):
            selected_cls = np.random.permutation(label_list)[:num_class_selected]
        else:
            selected_cls = label_list
        sample_function_params = {
            "dataset": test_set,
            "num_sample": feature_sample_size,
            "model": model_last_epoch,
            "criterion": criterion,
            "selected_labels": selected_cls,
            "seed": sub_seed,
        }
        sample_data_idx = sample_function(**sample_function_params)
        sample_dataset = torch.utils.data.Subset(test_set, sample_data_idx)
        sample_data_loader = torch.utils.data.DataLoader(
            sample_dataset, batch_size=100, shuffle=False, num_workers=2)

        ### Extract features
        for epoch_idx in range(len(epoch_list)):
            epoch = epoch_list[epoch_idx]
            print(f"Extracting epoch {epoch}. [{epoch_idx} / {len(epoch_list)}] epoch...")
            model_filepath = model_checkpoint_pattern.format(**model_checkpoint_filename_params,
                                            epoch=epoch)
            model = pipeline_utils.load_model_from_path(base_model, model_filepath)
            features = extract_features(model, sample_data_loader, layers)
            if do_save_feature:
                feature_filename_params =   {"save_folder": save_folder,
                            "mode": extraction_mode,
                            "filename_base": feature_filename_base,
                            "epoch": epoch,
                            }
                save_features(features, filename_config, feature_filename_params)
            if do_run_analysis:
                # Run feature analysis
                analysis_results = {}
                for layer in features:
                    feature_data = prepare_data_for_manifold_analysis(features[layer])
                    try:
                        analysis_result = gcmc.manifold_analysis(feature_data, backend="numpy")
                        analysis_results[layer] = analysis_result
                    except:
                        try:
                            feature_data += np.random.randn(*feature_data.shape)*1e-6
                            analysis_result = gcmc.manifold_analysis(feature_data, backend="numpy")
                            analysis_results[layer] = analysis_result
                        except Exception as e:
                            print(f"Run to error with params {params} at epoch {epoch} layer {layer} \n error msg {e}")
                # Save analysis results
                result_filename_params =   {"save_folder": save_folder,
                                "mode": analysis_mode,
                                "filename_base": analysis_filename_base,
                                "epoch": epoch,
                                }
                pipeline_utils.save_analysis_result(analysis_results, filename_config, result_filename_params, "analysis_result")
