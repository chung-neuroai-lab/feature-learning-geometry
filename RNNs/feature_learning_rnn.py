# Code adapted from https://github.com/Helena-Yuhan-Liu/BioRNN_RichLazy
import torch
import numpy as np
import os
import pickle as pkl
import pandas as pd
import argparse
import random
import gcmc
from gcmc.contrib import gcmc_analysis_dataframe
import neurogym as ngym


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define RNN 
# Code to setup RNN is adapted from https://github.com/gyyang/nn-brain/blob/master/RNN%2BDynamicalSystemAnalysis.ipynb

class CTRNN(torch.nn.Module):
    """Continuous-time RNN.

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        self.input2h = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        pre_activation = self.input2h(input) + self.h2h(hidden) 
        h_new = torch.relu(hidden * self.oneminusalpha +
                           pre_activation * self.alpha)
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, hidden


class Net(torch.nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)
        self.fc = torch.nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        out = self.fc(rnn_activity)
        return out, rnn_activity

def make_Xtot(features, labels, num_sample=50, eps=1e-6, add_noise=True):
    unique_labels = np.unique(labels)
    X_tot = np.stack([features[labels== i, :][:num_sample, :].T for i in unique_labels])
    if add_noise:
        X_tot += np.random.randn(*X_tot.shape)*eps
    return X_tot

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(0)

def setup_task(task_name):
    ### Setup task

    t_mult = 1
    # Environment  
    batch_size = 32
    seq_len = 100
    if task_name == '2AF':
        task = 'PerceptualDecisionMaking-v0'
        timing = {
            'fixation': 0*t_mult,
            'stimulus': 700*t_mult,
            'delay': 0*t_mult,
            'decision': 100*t_mult}
        seq_len = 8*t_mult
        kwargs = {'dt': 100, 'timing': timing}
        # Make supervised dataset
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=batch_size,
                            seq_len=seq_len)
    elif task_name == 'DMS':
        task = 'DelayMatchSample-v0'
        seq_len = 8*t_mult
        timing = {
            'fixation': 0*t_mult,
            'sample': 100*t_mult,
            'delay': 500*t_mult,
            'test': 100*t_mult,
            'decision': 100*t_mult}
        kwargs = {'dt': 100, 'timing': timing} 
        
        # Make supervised dataset
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=batch_size,
                            seq_len=seq_len)

    elif task_name == 'CXT':
        task = 'ContextDecisionMaking-v0'
        seq_len = 8*t_mult
        timing = {
            'fixation': 0*t_mult,
            # 'target': 350,
            'stimulus': 200*t_mult,
            'delay': 500*t_mult,
            'decision': 100*t_mult}
        kwargs = {'dt': 100, 'timing': timing} 
        
        # Make supervised dataset
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=batch_size,
                            seq_len=seq_len)

    # A sample environment from dataset
    env = dataset.env
    # Visualize the environment with 2 sample trials
    # _ = ngym.utils.plot_env(env, num_trials=2)
    return {"dataset": dataset, "env": env, "batch_size": batch_size, "seq_len": seq_len}

def get_var_list(hidden_size):
    var_list = np.array([1, 5, 10, 25, 50, 100, 200, 300])
    return var_list[var_list <= hidden_size]

def setup_weight_rank(W, rank):
    U_, S_, VT_ = np.linalg.svd(W)
    W_rank = U_[:,:rank] @ np.diag(S_[:rank]) @ VT_[:rank, :]
    return W_rank

def get_capacity(activity, labels, var, seed):
    X_tot = make_Xtot(activity.squeeze(), labels)
    try:
        capacity_res = gcmc_analysis_dataframe(X_tot, seed=seed, analysis_type='ONE_VERSUS_REST_ALL', preprocess_type='center', scale=0)
    except Exception as exception:
        print(f"Cannot run GCMC at var {var} seed {seed}. Exception: {exception}")
        capacity_res = None
    return capacity_res, X_tot

def get_accuracy(output, labels):
    pred = np.argmax(output, axis=2).flatten()
    acc = (pred == labels).sum() / len(labels)
    return acc

def get_rep_sim(activity0, activity):
    KR0 = activity0[-1,:,:] @ activity0[-1,:,:].T # (b,j) @ (j,b) -> (b,b)
    KR = activity[-1,:,:] @ activity[-1,:,:].T # (b,j) @ (j,b) -> (b,b)
    rep_sim = np.sum(KR*KR0) / np.linalg.norm(KR0) / np.linalg.norm(KR)
    return rep_sim

def compute_kernel(net, output, seq_len=8, batch_size=32, output_size=3):
    for t in range(seq_len): 
        for b in range(batch_size):
            for k in range(output_size): # output is the inner dim
                df_1 = torch.unsqueeze(torch.autograd.grad(output[t,b,k], \
                                            net.rnn.h2h.weight, retain_graph=True)[0], dim=0).cpu() # focus on recurrent weight
                if (b==0) and (k==0) and (t==0):
                    df = df_1
                else:
                    df = torch.cat((df, df_1), dim=0)
    kernel = torch.einsum('bij,aij->ba', df, df)
    return kernel

def run_pipeline(params_filename):     
    from mpi4py import MPI
    import os
    import sys
    import json

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
    params = json.load(open(params_filename, "r"))

    # get parameter for this job (in the parameter list json file)
    job_idx = rank + (num_threads * task_id)
    if job_idx > len(params) - 1:
        print(f"job index {job_idx} is out of range for params file with length {len(params)} at path {params_filename}")
        sys.exit(0)
    
    # run Python function depends on the param mode
    this_param = params[job_idx]
    run_main(this_param)

def run_main(params):
    ## Setup arguments
    parser_str = " ".join([f"--{key}={val}" for (key, val) in params.items()])
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_iter', default=10000, type=int, help='number of training iter')
    parser.add_argument('--print_step', default=100, type=int, help='frequency of saving data')
    parser.add_argument('--task', default='CXT', type=str, choices=['2AF', 'DMS', 'CXT'], help='task')
    parser.add_argument('--learning_rate0', default=0.003, type=float, help='base learning rate')
    parser.add_argument('--W0sig', default=1.25, type=float, help='relevant only if var_name=rr or kap2, std for the starting W init')
    parser.add_argument('--hidden_size', default=300, type=int, help='number of hidden units')
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--test_data_folder', default='', type=str, help="Folder that store test data")
    parser.add_argument('--save_folder', default='', type=str, help="Folder to save data")

    args = parser.parse_args(parser_str.split())
    print(args)
    seed = args.seed
    test_data_folder = args.test_data_folder
    save_folder = args.save_folder
    task = args.task
    set_seed(seed)
    # Set up task and weight
    task_dict = setup_task(task)
    dataset, env, batch_size, seq_len = task_dict["dataset"], task_dict["env"], task_dict["batch_size"], task_dict["seq_len"]
    W0_Gauss = args.W0sig*np.random.randn(args.hidden_size, args.hidden_size)/np.sqrt(args.hidden_size)

    # Set up test data
    test_data = pkl.load(open(os.path.join(test_data_folder, f"task_{args.task}.pkl"), "rb"))
    inputs0, labels0 = test_data["input"], test_data["labels"]
    inputs0 = torch.from_numpy(inputs0).type(torch.float)
    labels0_pt = torch.from_numpy(labels0.flatten()).type(torch.long)
    decision_labels = labels0[env.start_ind['decision']:env.end_ind['decision']].flatten()
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    dt = env.dt

    var_list = get_var_list(args.hidden_size)
    lr_list = [args.learning_rate0] #[0.001, 0.003, 0.01] 
    criterion = torch.nn.CrossEntropyLoss()

    running_loss = 0
    print_step = args.print_step
    hidden_size = args.hidden_size

    result_list = []
    for var in var_list:
        print(f"### {var=}")
        # Loop through hyperparameters 
        for lr in range(len(lr_list)): 
            # Instantiate the network and print information
            net = Net(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dt=dt)
            net = net.to(DEVICE)
            
            # set up weight rank here
            W0new = setup_weight_rank(W0_Gauss, var)

            # normalize
            W0new = W0new / np.linalg.norm(W0new) * np.linalg.norm(W0_Gauss) # by norm                          
            net.rnn.h2h.weight.data.copy_(torch.from_numpy(W0new).type(torch.float).to(DEVICE))
                                        
            # Optimizer TODO: May try Adam here?
            n_iter = args.n_iter
            optimizer = torch.optim.SGD(net.parameters(), lr=lr_list[lr], momentum=0.9) # default

            ## Storing initial stuff
            Wr_0 = net.rnn.h2h.weight.detach().cpu().numpy().copy()                  
                
            output0, activity0 = net(inputs0.to(DEVICE))
            loss_0 = criterion(output0.view(-1, output_size), labels0_pt.to(DEVICE)).item()
            activity0_ = activity0[env.start_ind['decision']:env.end_ind['decision']].detach().cpu().numpy().copy()
            this_capacity_res0, X_tot0 = get_capacity(activity0_, decision_labels, var, seed)

            # Get accuracy
            output0_ = output0[env.start_ind['decision']:env.end_ind['decision']].detach().cpu().numpy()
            acc_0 = get_accuracy(output0_, decision_labels)

            # Get K0
            K0 = compute_kernel(net, output0, seq_len=seq_len, batch_size=batch_size, output_size=output_size)

            ### start training ###
            loss_list = []
            iter_list = []
            for i in range(n_iter):
                inputs, labels_ = dataset()
                inputs = torch.from_numpy(inputs).type(torch.float)

                # in your training loop:
                optimizer.zero_grad()   # zero the gradient buffers            
                
                output, activity = net(inputs.to(DEVICE))
                labels = torch.from_numpy(labels_.flatten()).type(torch.long).to(DEVICE)
                output = output.view(-1, output_size)     
                loss = criterion(output, labels)
                
                loss.backward()
                optimizer.step()    # Does the update
                running_loss += loss.item()

                if i % print_step == 0: # (print_step - 1):
                    if i != 0:
                        running_loss /= print_step
                    print(f"Step {i}, {running_loss=:.4f}")
                    loss_list.append(running_loss)
                    iter_list.append(i)
                    running_loss = 0
            ### End training ###
        
            # View weights
            Wr = net.rnn.h2h.weight.detach().cpu().numpy()
            delta_w = np.linalg.norm(Wr-Wr_0)

            ### get the linearity measures ###
            output, activity = net(inputs0.to(DEVICE))
            activity_ = activity[env.start_ind['decision']:env.end_ind['decision']].detach().cpu().numpy().copy()
            # get loss
            loss_t = criterion(output.view(-1, output_size), labels0_pt.to(DEVICE)).item()

            # get capacity
            this_capacity_res, X_tot = get_capacity(activity_, decision_labels, var, seed)
            
            # Get accuracy
            output_ = output[env.start_ind['decision']:env.end_ind['decision']].detach().cpu().numpy()
            acc_t = get_accuracy(output_, decision_labels)

            # Get sign similarity
            sign_sim = np.sum(np.sign(activity_)==np.sign(activity0_))/activity_.size
                        
            # Get rep similarity, based on hidden activity at decision time 
            rep_sim = get_rep_sim(activity0_, activity_)
                        
            # Get Kf
            Kf = compute_kernel(net, output, seq_len=seq_len, batch_size=batch_size, output_size=output_size)
            kernel_alignment = (torch.sum(Kf*K0) / torch.norm(Kf) / torch.norm(K0)).detach().cpu().numpy().copy()     
            result_list.append({
                "lr": lr_list[lr],
                "weight_rank": var,
                "delta_W": delta_w,
                "capacity_0": this_capacity_res0,
                "capacity_t": this_capacity_res,
                "sign_sim": sign_sim,
                "rep_sim": rep_sim,
                "kernel_alignment": kernel_alignment,
                "seed": seed,
                "acc_0": acc_0,
                "acc_t": acc_t,
                "loss_0": loss_0,
                "loss_t": loss_t,
                "loss_list": loss_list,
                "iter_list": iter_list,
                "task": args.task,
                "X_tot": X_tot,
                "X_tot0": X_tot0,
            })
            print(f"{var=}, {acc_0=}, {acc_t=}")
    save_file_path = os.path.join(save_folder, f"weight_rank_task_{task}_seed_{seed}.pkl")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    pkl.dump(result_list, open(save_file_path, "wb"))
    print(f"Save file to {save_file_path=}")

if __name__ == "__main__":
    pipeline_parser = argparse.ArgumentParser()
    pipeline_parser.add_argument('-p','--params_filename', required=True, type=str, help="Path to parameter json file")
    pipeline_args = pipeline_parser.parse_args()
    run_pipeline(pipeline_args.params_filename)
