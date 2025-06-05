import abc
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random

def flatten_list(list_of_list):
    return [elt for this_list in list_of_list for elt in this_list]

def get_dataset_obj_by_name(dataset_name, root_folder):
    dataset_name_to_obj = {
        "CIFAR-10": CIFAR10Dataset,
        "CIFAR-100": CIFAR100Dataset,
    }
    if dataset_name not in dataset_name_to_obj:
        raise ValueError(f"Dataset {dataset_name} is not in supported dataset list {dataset_name_to_obj.keys()}!")
    else:
        dataset_obj = dataset_name_to_obj[dataset_name](dataset_name, root_folder)
    return dataset_obj

def get_sample_function_by_name(sample_method):
    sample_method_dict = {
        "random": random_sample,
        "minLoss": sample_minloss,
    }
    if sample_method not in sample_method_dict:
        raise ValueError(f"Sample method {sample_method} is not in supported sample method list {sample_method.keys()}!")
    else:
        return sample_method_dict[sample_method]

def random_sample(dataset, num_sample, model, criterion, selected_labels=None, seed=0):

    # TODO: This function is the naive implementation and can be implemented in a faster way
    # TODO: seed the whole module, not only this function
    random.seed(seed)
    if selected_labels is None:
        unique_values = list(set(dataset.targets))
    else:
        unique_values = selected_labels
    val_to_idx = {val: [] for val in unique_values}
    for idx in range(len(dataset)):
        val = dataset[idx][1]
        if val in val_to_idx:
            val_to_idx[val].append(idx)
    sample_map = {key: random.sample(val, k=num_sample) for (key, val) in val_to_idx.items()}
    return flatten_list(list(sample_map.values()))

def create_few_shot_dataset(train_set, test_set, num_examples, num_classes, seed=0):
    # sample `num_classes`, from each of these classes, sample `num_examples` for training set
    # keep all examples for test set
    # TODO: This function is the naive implementation and can be implemented in a faster way
    np.random.seed(seed)
    selected_classes = np.random.permutation(list(set(train_set.targets)))[:num_classes]
    train_idx = random_sample(train_set, num_examples, None, None, selected_labels=selected_classes, seed=seed)
    test_idx = [i for i in range(len(test_set)) if test_set.targets[i] in selected_classes]
    
    fewshot_trainset = torch.utils.data.Subset(train_set, train_idx)
    fewshot_testset = torch.utils.data.Subset(test_set, test_idx)
    fewshot_trainloader = torch.utils.data.DataLoader(
        fewshot_trainset, batch_size=100, shuffle=True)
    fewshot_testloader = torch.utils.data.DataLoader(
        fewshot_testset, batch_size=100, shuffle=False)

    return {"train_set": fewshot_trainset, "test_set": fewshot_testset, "train_loader": fewshot_trainloader, "test_loader": fewshot_testloader, "selected_classes": selected_classes}

def sample_minloss(dataset, num_sample, model, criterion):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128)
    unique_values = list(set(dataset.targets))
    loss_dict = {"target": [],
                "loss": [], 
                "idx": []}
    for input, target, idx in data_loader:
        with torch.no_grad():
            logit = model(input)
            if criterion.__class__.__name__ == "MSELoss":
                target_ = target.unsqueeze(1)
                targets_embed = torch.zeros(target_.size(0), 10)
                targets_embed.scatter_(1, target_, 1)
                loss = criterion(input=logit, target=targets_embed)
                loss = torch.mean(loss, dim=-1) # average across k classes
            else: # cross entropy loss
                loss = criterion(input=logit, target=target)
        loss_dict["loss"].extend([i.item() for i in loss])
        loss_dict["target"].extend([i.item() for i in target])
        loss_dict["idx"].extend([i.item() for i in idx])

    target_to_loss_dict = {val: [] for val in unique_values}
    for i in range(len(loss_dict["target"])):
        target_to_loss_dict[loss_dict["target"][i]].append((loss_dict["loss"][i], loss_dict["idx"][i]))

    sorted_loss_dict = {
        key: sorted(val, key=lambda x: x[0]) for (key, val) in target_to_loss_dict.items()
    }
    chosen_loss_dict = {key: [val[i][1] for i in range(num_sample)] for (key, val) in sorted_loss_dict.items()}
    return flatten_list(list(chosen_loss_dict.values()))
    
def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

CIFAR10WithIndices = dataset_with_indices(torchvision.datasets.CIFAR10)
CIFAR100WithIndices = dataset_with_indices(torchvision.datasets.CIFAR100)

class BaseDataset(metaclass=abc.ABCMeta):
    def __init__(self, dataset_name, root_path, batch_size=128, num_worker=2):
        self.dataset_name = dataset_name
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_worker = num_worker
    
    @abc.abstractmethod
    def get_traindata(self):
        return

    @abc.abstractmethod
    def get_testdata(self):
        return

    def get_dataset(self):
        return {**self.get_traindata(), **self.get_testdata()}
    
    @staticmethod
    def get_normalization_params(dataset, num_sample=50000):
        import numpy as np
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=num_sample, shuffle=False)
        sample_dataset = iter(data_loader).__next__()[0].numpy() # get first batch of dataset
        mean = np.mean(sample_dataset, axis=(0,2,3)) # get mean for each channel (across batch, height, and width)
        std = np.std(sample_dataset, axis=(0,2,3))
        return {"mean": mean, "std": std}
    

class CIFAR10Dataset(BaseDataset):
    def __init__(self, dataset_name, root_path, batch_size=100, num_worker=2):
        super().__init__(dataset_name, root_path, batch_size=batch_size, num_worker=num_worker)
        # Load data to calculate mean and std for normalization transformation
        train_set = CIFAR10WithIndices(
            root=self.root_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        self.normalization_params = CIFAR10Dataset.get_normalization_params(train_set)
    
    def get_traindata(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.normalization_params["mean"], self.normalization_params["std"])
        ])
        train_set = CIFAR10WithIndices(
            root=self.root_path, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
        return {"train_set": train_set, "train_loader": train_loader}

    def get_testdata(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.normalization_params["mean"], self.normalization_params["std"]),
        ])
        test_set = CIFAR10WithIndices(
            root=self.root_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker)
        return {"test_set": test_set, "test_loader": test_loader}


class CIFAR100Dataset(BaseDataset):
    def __init__(self, dataset_name, root_path, batch_size=128, num_worker=2):
        super().__init__(dataset_name, root_path, batch_size=batch_size, num_worker=num_worker)
        # Load data to calculate mean and std for normalization transformation
        train_set = CIFAR100WithIndices(
            root=self.root_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        self.normalization_params = CIFAR100Dataset.get_normalization_params(train_set)
    
    def get_traindata(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.normalization_params["mean"], self.normalization_params["std"])
        ])
        train_set = CIFAR100WithIndices(
            root=self.root_path, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
        return {"train_set": train_set, "train_loader": train_loader}

    def get_testdata(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.normalization_params["mean"], self.normalization_params["std"]),
        ])
        test_set = CIFAR100WithIndices(
            root=self.root_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker)
        return {"test_set": test_set, "test_loader": test_loader}
