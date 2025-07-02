import torch
import os
import pickle
import pipeline_dataset
import pipeline_utils
import torch.nn.functional as F
import numpy as np
import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(params, resume=False, filename_config=None, filename_params=None):

    # Works for `resnet`, `alexnet`
    # TODO: Parameterize optimizer and scheduler kwargs
    # TODO: Add k for model width
    model_name, optimizer_name, learning_rate = params["model"], params["optim"], params["lr"]
    if "decay" in params:
        if params["alpha"] == 1:
            weight_decay = params["decay"]
        else:
            weight_decay = params["decay"] / (params["alpha"]**2) # need to adjust with `alpha` because we use loss function * (1/alpha**2)
    else:
        weight_decay = 0.0
    print(f"Raw weight decay: {params['decay']}, alpha: {params['alpha']}, effective weight decay: {weight_decay}")

    print(f"Get model name: {model_name}")
    model = pipeline_utils.load_base_model(model_name)
    if resume:
        assert filename_config is not None, f"filename_config is {filename_config}. Must input filename_config if resume model training!"
        assert filename_params is not None, f"filename_params is {filename_params}. Must input filename_params if resume model training!"
        checkpoint_filepath = filename_config["model_checkpoint"].format(**filename_params)
        model = pipeline_utils.load_model_from_path(model, checkpoint_filepath)

    model = model.to(DEVICE)
    if DEVICE == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    
    # Get optimizer and scheduler
    # TODO: Need to switch to `AdamW` if want to use weight decay with Adam
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = None
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=0.9, weight_decay=weight_decay)
        scheduler = None
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported. List of supported optimizer [Adam, SGD]")

    return model, optimizer, scheduler

def get_dataset(dataset_name, root_folder):
        dataset_obj = pipeline_dataset.get_dataset_obj_by_name(dataset_name, root_folder)
        return dataset_obj.get_dataset()

def run_train(model, model_0, criterion, optimizer, trainloader, params):
    model.train()
    train_loss_scaled = 0
    train_loss = 0
    correct = 0
    total = 0
    alpha = params["alpha"]
    for batch_idx, (inputs, targets, idx) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        if params["precision"] == "double":
            inputs = inputs.double()
        optimizer.zero_grad()
        outputs_ = model(inputs)
        outputs_0 = model_0(inputs)
        outputs = outputs_ - outputs_0
        if params["criterion"] == 'ce':
            loss_scaled = criterion(alpha*outputs, targets)/alpha**2
            loss = criterion(outputs, targets)
        elif params["criterion"] == 'mse':
            targets_=targets.unsqueeze(1)
            targets_embed=torch.zeros(targets_.size(0),10).cuda()
            targets_embed.scatter_(1, targets_, 1)
            loss_scaled = criterion(outputs, targets_embed/alpha)
            loss = criterion(outputs, targets_embed)
        else:
            raise ValueError(f"Loss must be crossEntropy or mse, but is {params['criterion']}")
        loss_scaled.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip grad to avoid grad explode
        optimizer.step()

        train_loss_scaled += loss_scaled.item()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if params["useStep"]: # if using gradient step, then break and not train the whole epoch
            break
    
    train_loss_scaled = train_loss_scaled/(batch_idx+1)
    train_loss = train_loss/(batch_idx+1)
    train_acc = 100.*correct/total
    print(f'[Train] Log Loss Scaled: {np.log(train_loss_scaled)} | Log Loss: {np.log(train_loss)} | Acc: {train_acc: .3f}%% ({correct}/{total})')
    return train_loss, train_acc, train_loss_scaled

@torch.no_grad
def run_test(model, model_0, model_clone, criterion, testloader, params, mode="Test"):
    global proportion_lazy
    test_loss = 0
    test_loss_scaled = 0
    correct = 0
    total = 0
    alpha = params["alpha"]

    for batch_idx, (inputs, targets, idx) in enumerate(testloader):
        inputs, targets = inputs.to(DEVICE).float(), targets.to(DEVICE)
        if params["precision"] == "double":
            inputs=inputs.double()
        if mode == "Test":
            net_activation(model.module, model_clone, inputs, precision=params["precision"]) # calculate `proportion_lazy`
        outputs_ = model(inputs)
        outputs_0 = model_0(inputs)
        outputs = outputs_ - outputs_0

        if params["criterion"] == "ce":
            loss = criterion(outputs, targets)
            loss_scaled = criterion(alpha * outputs, targets) / alpha ** 2
        elif params["criterion"] == 'mse':
            targets_ = targets.unsqueeze(1)
            targets_embed = torch.zeros(targets_.size(0), 10).cuda()
            targets_embed.scatter_(1, targets_, 1)
            loss = criterion(outputs, targets_embed)
            loss_scaled = criterion(outputs, targets_embed / alpha)
        else:
            raise ValueError(f"Loss must be crossEntropy or mse, but is {params['criterion']}")

        test_loss_scaled += loss_scaled.item()
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    proportion_lazy = [proportion_lazy_i / total for proportion_lazy_i in proportion_lazy]
    test_loss = test_loss/(batch_idx+1)
    test_loss_scaled = test_loss_scaled / (batch_idx+1)
    test_acc = 100.*correct/total
    print(f'[{mode}] Log Loss: {np.log(test_loss)} | Log Loss scaled: {np.log(test_loss_scaled)} | Acc: {test_acc: .3f}%% ({correct}/{total})')    
    return test_loss, test_acc, test_loss_scaled

def save_checkpoint(model, filename_config, filename_params):

    state = {'model': model.state_dict()}
    filepath = filename_config["model_checkpoint"].format(**filename_params)
    parent_folder = os.path.dirname(filepath)
    if not os.path.isdir(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)
    torch.save(state, filepath)
    print(f'Saving epoch {filename_params["epoch"]} to file path {filepath}...')

def save_training_state(state_list, filename_config, filename_params):
    filepath = filename_config["training_state"].format(**filename_params)
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

##### Additional functions to train lazy CNNs

def weights_init(m, gain=1.0):
    # Initialize the weight with Xavier initialization and set bias to 0
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data,gain=gain)
        m.bias.data.zero_()
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        m.bias.data.zero_()

####### Set up hook. TODO: Refactor hook
def hook_extract_relu(module, input, out, precision='double'):
    global stack_hook
    p = out > 0
    if precision =='float':
        p = p.float()
    else:
        p=p.double()
    stack_hook.append(p)

def hook_extract_maxpool(module, inp, outp, precision='double'):
    global stack_hook
    global pooling_layer
    global unpooling_layer
    inp = inp[0]

    _,idx=pooling_layer(inp)
    out = unpooling_layer(outp,idx)
    p = out > 0

    if precision == 'float':
        p = p.float()
    else:
        p = p.double()
    stack_hook.append(p)

def hook_extract_basicblock(module, inp, outp, precision='double'):
    global stack_hook
    inp = inp[0]
    a = F.relu(module.conv1(inp))
    p = a > 0
    if precision == 'float':
        p = p.float()
    else:
        p = p.double()

    q = outp>0
    if precision == 'float':
        q = q.float()
    else:
        q = q.double()
    stack_hook.append([p,q])

def register_hook(model_clone):
    for i in range(len(model_clone.features)):
        if model_clone.features[i].__class__.__name__=='ReLU':
            model_clone.features[i].register_forward_hook(hook_extract_relu)
        elif model_clone.features[i].__class__.__name__ == 'MaxPool2d':
            model_clone.features[i].register_forward_hook(hook_extract_maxpool)
        elif model_clone.features[i].__class__.__name__ == 'BasicBlock':
            model_clone.features[i].register_forward_hook(hook_extract_basicblock)

@torch.no_grad
def net_activation(model, model_clone, x, precision='double'):
    """Extract model activation and compute `proportion_lazy` (activation stability)
    """
    global stack_hook
    global proportion_lazy
    z = x.clone()
    stack_hook = []
    model_clone(x)

    j = 0
    for i in range(len(model.features)):
        if model.features[i].__class__.__name__ == 'ReLU':
            p = stack_hook[j]
            z = model.features[i](z)
            p_=z>0
            if precision == 'float':
                p_ = p_.float()
            else:
                p_ = p_.double()
            num_features = float(np.prod(list(p.size()[1:])))
            proportion_lazy[j]+= float(torch.sum(p_ == p)) / num_features
            j = j + 1
        elif model.features[i].__class__.__name__ == 'MaxPool2d':
            p = stack_hook[j]
            z_ = z * p
            z_, _ = pooling_layer(z_)
            z = model.features[i](z)
            num_features = float(np.prod(list(z.size()[1:])))
            proportion_lazy[j]+=float(torch.sum(z == z_)) / num_features
            j = j + 1
        elif model.features[i].__class__.__name__ == 'BasicBlock':
            p, q = stack_hook[j]
            z = model.features[i](z)
            q_ = z>0
            if precision == 'float':
                q_ = q_.float()
            else:
                q_ = q_.double()
            num_features = float(np.prod(list(q.size()[1:])))
            proportion_lazy[j] += float(torch.sum(q_ == q)) / num_features
            j = j + 1
        else:
            z = model.features[i](z)
    z = z.view(z.size(0), -1)
    return z

def compute_weight_norm(model):
    weight_norm = 0.0
    for p in model.parameters():
        param_weight_norm = p.data.norm(2)
        weight_norm += param_weight_norm.item() **2
    weight_norm = weight_norm ** (1. / 2)
    return weight_norm

def compute_grad_norm(model):
    grad_norm = 0.0
    for p in model.parameters():
        param_grad_norm = p.grad.detach().data.norm(2)
        grad_norm += param_grad_norm.item() ** 2
    grad_norm = grad_norm ** (1. / 2)
    return grad_norm

def compute_weight_norm_diff(model_0, model_1):
    model_0_params = list(model_0.parameters())
    model_1_params = list(model_1.parameters())
    diff_norm = 0
    if len(model_0_params) != len(model_1_params):
        raise ValueError(f"Models have different number of parameters! Model 0 has {len(model_0_params)} params, and model 1 has {len(model_1_params)} params!")
    else:
        for i in range(len(model_0_params)):
            model_0_param, model_1_param = model_0_params[i], model_1_params[i]
            weight_diff = model_0_param.data - model_1_param.data
            weight_diff_norm = weight_diff.norm(2)
            diff_norm += weight_diff_norm.item() ** 2
        diff_norm = diff_norm ** (1. / 2)
        return diff_norm

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(0)

def main(params):
    import re
    import glob
    import copy
    import torch.nn as nn
    
    print(f"=== Process: Train the model ===")
    print(f"Parameters: {params}")

    if params["precision"] == "float":
        torch.set_default_dtype(torch.float32)
    elif params["precision"] == "double": # for lazy training, must use double precision to avoid gradient overflow
        torch.set_default_dtype(torch.float64)

    ### Output folder and filename
    save_folder = params["save_folder"]
    seed = params["seed"] if "seed" in params else 0
    set_seed(seed)
    filename_config = pipeline_utils.get_config()["filename_config"]
    filename_base = pipeline_utils.generate_filename(params, params["mode"])
    filename_params_base = {"save_folder": save_folder, "filename_base": filename_base, "mode": params["mode"]}
    num_epoch = params["num_epoch"]
    use_step = bool(params["useStep"])
    epoch_list = pipeline_utils.generate_epoch_list(num_epoch, use_step=use_step)

    ### Load dataset
    print("=== Loading dataset ====")
    dataset_name = params["task"]
    input_folder = filename_config["input_data"]
    dataset_dict = get_dataset(dataset_name, input_folder)
    train_set, test_set, train_loader, test_loader = dataset_dict["train_set"], dataset_dict["test_set"], dataset_dict["train_loader"], dataset_dict["test_loader"]

    ### Load model
    if "resume" in params and params["resume"]:
        # Find the largest epoch saved model
        checkpoint_glob_pattern = filename_config["model_checkpoint"].format(**filename_params_base, epoch="[0-9]*")
        checkpoint_regex_pattern = filename_config["model_checkpoint"].format(**filename_params_base, epoch="([0-9]+)")
        checkpoint_filenames = glob.glob(checkpoint_glob_pattern)
        largest_epoch = max([int(re.search(checkpoint_regex_pattern, filename).group(1)) for filename in checkpoint_filenames])
        model, optimizer, scheduler = get_model(params, #params["model"], params["optim"], params["lr"],
                                                filename_config=filename_config,
                                                filename_params={**filename_params_base, "epoch": largest_epoch}) 
        epoch_start = largest_epoch + 1
        print(f'=== Resume from epoch {epoch_start} ===')
    else:
        epoch_start = 0
        state_list = []
        model, optimizer, scheduler = get_model(params) #params["model"], params["optim"], params["lr"])
    criterion = pipeline_utils.get_criterion(params["criterion"])

    model = model.apply(weights_init)
    model_0 = copy.deepcopy(model)
    model_clone = copy.deepcopy(model)

    # Add feature hook to calculate `proportion_lazy`
    global pooling_layer; global unpooling_layer; global proportion_lazy; global stack_hook
    pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
    unpooling_layer = nn.MaxUnpool2d(kernel_size=2, stride=2)
    register_hook(model_clone.module)
    
    # Populate hook
    stack_hook = []
    x=torch.randn(1,3,32,32).cuda()
    if params["precision"] == "double":
        x=x.double()
    model_clone(x)
    proportion_lazy = [0] * len(stack_hook)
    del x

    ### Train, test, save model
    initial_weight_norm = None
    for epoch in range(epoch_start, num_epoch):
        for i in range(len(stack_hook)):
            stack_hook[i]=None

        proportion_lazy = [0] * len(stack_hook)
        if epoch == 0:
            print(f"Untrained model")
            train_loss, train_acc, train_loss_scaled = run_test(model, model_0, model_clone, criterion, train_loader, params, mode="Train")
            test_loss, test_acc, test_loss_scaled = run_test(model, model_0, model_clone, criterion, test_loader, params)
            weight_norm = compute_weight_norm(model)
            initial_weight_norm = weight_norm
            grad_norm = None
        else:
            print(f"Running epoch {epoch}")
            if params["useStep"]:
                if epoch % 500 == 0:
                    epoch_lr = epoch // 500
                    this_lr = params["lr"] /(1.0+100.0*(epoch_lr-1)/300.0)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = this_lr
            else:
                this_lr = params["lr"] /(1.0+100.0*(epoch-1)/300.0)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = this_lr
            train_loss, train_acc, train_loss_scaled = run_train(model, model_0, criterion, optimizer, train_loader, params)
            test_loss, test_acc, test_loss_scaled = run_test(model, model_0, model_clone, criterion, test_loader, params)
            weight_norm = compute_weight_norm(model)
            grad_norm = compute_grad_norm(model)
        weight_norm_diff = compute_weight_norm_diff(model, model_0)

        state = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_scaled": train_loss_scaled,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_loss_scaled": test_loss_scaled,
            "test_acc": test_acc,
            "proportion_lazy": proportion_lazy,
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
            "initial_weight_norm": initial_weight_norm,
            "weight_norm_diff": weight_norm_diff,
        }
        state_list.append(state)
        print(proportion_lazy)
        
        if epoch in epoch_list:
            checkpoint_filename_params = {**filename_params_base, "epoch": epoch}
            save_checkpoint(model, filename_config, checkpoint_filename_params)
        if scheduler and epoch > 0:
            scheduler.step()
        
    # Save last epoch for resuming training
    checkpoint_filename_params = {**filename_params_base, "epoch": epoch}
    save_checkpoint(model, filename_config, checkpoint_filename_params)

    # Save epoch state (accuracy, loss, etc.)
    save_training_state(state_list, filename_config, filename_params_base)
