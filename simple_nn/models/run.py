import torch
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
import time, os, re
from tqdm import tqdm
from simple_nn.models import neural_network, loss, optimizers, logger
from functools import partial


#Main function that train neural network
def train(inputs, logfile):
    # Load data from 'train_pool', 'valid_pool', 'elem_list'
    elem_list=[]
    with open('elem_list', 'r') as f:
        for line in f.readlines():
            elem_list.append(line.strip())

    elem_num = dict()
    for e in elem_list:
        elem_num[e] = 0
    train_pool = []
    p=re.compile('[A-Z][a-z]?')
    with open('train_pool', 'r') as f:
        for line in f.readlines():
            train_pool.append(line.strip())
            system = line.split('_')[1].split('.')[0]
            els = p.findall(system)
            for e in els:
                elem_num[e] += 1
    elem_weights = dict()
    for e in elem_list:
        elem_weights[e] = 1/elem_num[e]

    valid_pool = []
    with open('valid_pool', 'r') as f:
        for line in f.readlines():
            valid_pool.append(line.strip())

    if inputs['neural_network']['double_precision']:
        torch.set_default_dtype(torch.float64)
    device = _get_torch_device(inputs)

    # Initialize weights of each elements
    init_weights = neural_network._initialize_weights(inputs, logfile, device, elem_list)

    # MAML start
    train_task_generator = TaskGenerator(train_pool, inputs['neural_network']['num_task'])
    valid_task_generator = TaskGenerator(valid_pool, len(valid_pool))
    start_time = time.time()
    for iter_idx in range(1, inputs['neural_network']['num_iter']+1):
        # Generate train tasks
        load_start_time = time.time()
        train_files, train_tasks = train_task_generator.generate_task(\
                inputs['neural_network']['num_support'], inputs['neural_network']['num_query'], device)
        load_time = time.time() - load_start_time

        # Outer loop start
        outer_loss_batch = []
        rmse_support_batch_init = []
        rmse_support_batch = []
        rmse_query_batch = []
        loss_grads_sum = dict()
        for el in elem_list:
            loss_grads_sum[el] = None

        start_time = time.time()
        for task_idx, task in enumerate(train_tasks):
            support, query = task
            atom_types = list(support[0].keys())
            atom_types.pop(atom_types.index('E'))

            # Run inner loop
            train_rmse_list, valid_loss, valid_rmse, val_loss_grads = inner_loop(inputs, logfile, task, atom_types, init_weights, device, cal_grads=True)
            rmse_support_batch_init.append(train_rmse_list[0])
            rmse_support_batch.append(train_rmse_list[-1])
            rmse_query_batch.append(valid_rmse)
            outer_loss_batch.append(valid_loss)

            # Update valid loss gradient
            for el in atom_types:
                if loss_grads_sum[el] == None:
                    loss_grads_sum[el] = val_loss_grads[el]
                else:
                    for i in range(len(loss_grads_sum[el])):
                        loss_grads_sum[el][i] += val_loss_grads[el][i]

        # Update initial weights
        for el in atom_types:
            for idx, key in enumerate(init_weights[el].keys()):
                #init_weights[el][key] -= inputs['neural_network']['outer_lr'] * elem_weights[el] * loss_grads_sum[el][idx] 
                init_weights[el][key] -= inputs['neural_network']['outer_lr'] * loss_grads_sum[el][idx] 

        outer_loss = np.mean(outer_loss_batch)
        rmse_support_init = np.mean(rmse_support_batch_init)
        rmse_support = np.mean(rmse_support_batch)
        rmse_query = np.mean(rmse_query_batch)
        total_time = time.time() - start_time

        logfile.write('Task systems:')
        for f_l in train_files:
            logfile.write('  %s'%(f_l.split('_')[1].split('.')[0]))
        logfile.write('\n')
        logfile.write(
            f'Iteration {iter_idx}: '
            f'loss: {outer_loss:.4e} | '
            f'support RMSE: '
            f'{rmse_support_init:.4e} > '
            f'{rmse_support:.4e} | '
            f'query RMSE: '
            f'{rmse_query:.4e} | '
            f'Task load time: '
            f'{load_time:.2f} | '
            f'Iteration time: '
            f'{total_time:.2f}\n'
        )
        logfile.flush()

        # Validation
        if iter_idx % inputs['neural_network']['valid_steps'] == 0:
            valid_files, valid_tasks = valid_task_generator.generate_task(\
                    inputs['neural_network']['num_support'], inputs['neural_network']['num_query'], device)
            for task_idx, task in enumerate(valid_tasks):
                support, query = task
                atom_types = list(support[0].keys())
                atom_types.pop(atom_types.index('E'))

                train_rmse_list, valid_loss, valid_rmse, val_loss_grads = inner_loop(inputs, logfile, task, atom_types, init_weights, device, cal_grads=False)
                rmse_support_batch.append(train_rmse_list[-1])
                rmse_query_batch.append(valid_rmse)
                outer_loss_batch.append(valid_loss)
                logfile.write(
                    f'Valid {valid_files[task_idx].split("_")[1].split(".")[0]}: '
                    f'loss: {valid_loss:.4e} | '
                    f'support RMSE: '
                    f'{train_rmse_list[-1]:.4e} | '
                    f'query RMSE: '
                    f'{valid_rmse:.4e}\n'
                )
            logfile.flush()

        if iter_idx % inputs['neural_network']['save_iter'] == 0:
            if not os.path.isdir('pot%s'%iter_idx):
                os.mkdir('pot%s'%iter_idx)
            model = neural_network._initialize_model(inputs, init_weights, logfile, device, ['O', 'Li', 'Na'])
            net = model.nets['O']
            for el in elem_list:
                net.load_state_dict(init_weights[el])
                write_lammps_potential('pot%s/potential_saved_%s'%(iter_idx, el), inputs, el, net)


    logfile.write(f"Elapsed time in training: {time.time()-start_time:10} s.\n")
    logfile.write("{}\n".format('-'*88))

def write_lammps_potential(filename, inputs, elem, net):
    FIL = open(filename, 'w')
    params = list()
    with open(inputs['params']) as fil:
        for line in fil:
            tmp = line.split()
            params += [list(map(float, tmp))]
    params = np.array(params)

    FIL.write('POT {} {}\n'.format(elem, np.max(params[:,3])))
    FIL.write('SYM {}\n'.format(len(params)))

    for ctem in params:
        if len(ctem) != 7:
            print(len(ctem))
            raise ValueError("params file must have lines with 7 columns.")

        FIL.write('{} {} {} {} {}\n'.\
            format(int(ctem[0]), ctem[3], ctem[4], ctem[5], ctem[6]))

    with open(inputs['params'], 'r') as f:
        tmp = f.readlines()
    input_dim = len(tmp) #open params read input number of symmetry functions
    FIL.write('scale1 {}\n'.format(' '.join(np.zeros(input_dim).astype(np.str))))
    FIL.write('scale2 {}\n'.format(' '.join(np.ones(input_dim).astype(np.str))))

    # An extra linear layer is used for PCA transformation.
    nodes   = list()
    weights = list()
    biases  = list()
    for n, i in net.lin.named_modules():
        if 'lin' in n:
            nodes.append(i.weight.size(0))
            weights.append(i.weight.detach().cpu().numpy())
            biases.append(i.bias.detach().cpu().numpy())
    nlayers = len(nodes)
    joffset = 0
    FIL.write('NET {} {}\n'.format(len(nodes)-1, ' '.join(map(str, nodes))))

    for j in range(nlayers):
        if j == nlayers-1:
            acti = 'linear'
        else:
            acti = inputs['neural_network']['acti_func']

        FIL.write('LAYER {} {}\n'.format(j+joffset, acti))

        for k in range(nodes[j + joffset]):
            FIL.write('w{} {}\n'.format(k, ' '.join(weights[j][k,:].astype(np.str))))
            FIL.write('b{} {}\n'.format(k, biases[j][k]))
    FIL.write('\n')
    FIL.close()

def inner_loop(inputs, logfile, task, atom_types, init_weights, device, cal_grads=True):
    support, query = task
    model = neural_network._initialize_model(inputs, init_weights, logfile, device, atom_types)
    #optimizer1 = optimizers._initialize_optimizer(inputs, model, [0, 1])
    #optimizer2 = optimizers._initialize_optimizer(inputs, model, [2, 3, 4, 5])
    #optimizer = [optimizer1, optimizer2]
    optimizer = optimizers._initialize_optimizer(inputs, model)
    criterion = torch.nn.MSELoss(reduction='none').to(device=device)

    # Make dataloader
    partial_collate = partial(my_collate, atom_types=atom_types, device=device,\
        scale_factor=None, pca=None,  pca_min_whiten_level=None,\
        use_force=False, use_stress=False)
    support_dataloader = torch.utils.data.DataLoader(\
        support, batch_size=inputs['neural_network']['batch_size'], shuffle=True, collate_fn=partial_collate,\
        num_workers=inputs['neural_network']['subprocesses'], pin_memory=False)
    query_dataloader = torch.utils.data.DataLoader(\
        query, batch_size=inputs['neural_network']['batch_size'], shuffle=True, collate_fn=partial_collate,\
        num_workers=inputs['neural_network']['subprocesses'], pin_memory=False)

    # Train using support set
    train_rmse_list, train_loss_list = train_model(inputs, atom_types, logfile, \
            model, optimizer, criterion, None, None, device, float('inf'), support_dataloader)

    # Valid using query set
    struct_labels = ['None']
    dtype = torch.get_default_dtype()
    non_block = False if (device == torch.device('cpu')) else True
    valid_epoch_result, valid_batch_loss = progress_epoch(inputs, atom_types, query_dataloader, struct_labels, model, optimizer, criterion, dtype, device, non_block, valid=True, atomic_e=False)
    valid_loss = valid_epoch_result['losses'].avg
    v_sum   = valid_epoch_result['e_err']['None'].sum
    v_count = valid_epoch_result['e_err']['None'].count
    valid_rmse = (v_sum / v_count) ** 0.5

    # Calculate gradient using valid loss
    optimizer.zero_grad()
    valid_batch_loss.backward()
    val_loss_grads = dict()
    if cal_grads == True:
        for el in atom_types:
            val_loss_grads[el] = []
            for g in model.nets[el].parameters():
                val_loss_grads[el].append(g.grad)

    return train_rmse_list, valid_loss, valid_rmse, val_loss_grads

def train_model(inputs, atom_types, logfile, model, optimizer, criterion, scale_factor, pca, device, best_loss, train_loader, atomic_e=False):
    struct_labels = ['None']
    dtype = torch.get_default_dtype()
    non_block = False if (device == torch.device('cpu')) else True

    max_len = len(train_loader)
    total_epoch = int(inputs['neural_network']['total_epoch'])
    total_iter = int(inputs['neural_network']['total_epoch'] * max_len)
    batch_size = 'full_batch' if inputs['neural_network']['full_batch'] else inputs['neural_network']['batch_size']
    best_epoch = 1

    if inputs['neural_network']['decay_rate']:
        scheduler = ExponentialLR(optimizer=optimizer, gamma=inputs['neural_network']['decay_rate'])

    # Train using support set
    train_rmse_list = []
    train_loss_list = []
    start_time = time.time()
    for epoch in tqdm(range(0, total_epoch+1), unit='epoch'):
        if epoch > 0:
            train_epoch_result, train_batch_loss = progress_epoch(inputs, atom_types, train_loader, struct_labels, model, optimizer, criterion, dtype, device, non_block, valid=False, atomic_e=atomic_e)
        train_epoch_result, train_batch_loss = progress_epoch(inputs, atom_types, train_loader, struct_labels, model, optimizer, criterion, dtype, device, non_block, valid=True, atomic_e=atomic_e)
        train_loss = train_epoch_result['losses'].avg
        t_sum   = train_epoch_result['e_err']['None'].sum
        t_count = train_epoch_result['e_err']['None'].count
        t_E_rmse = (t_sum / t_count) ** 0.5

        if epoch > 0 and inputs['neural_network']['decay_rate']:
            scheduler.step()

        train_rmse_list.append(t_E_rmse)
        train_loss_list.append(train_loss)

    total_time = time.time() - start_time

    return train_rmse_list, train_loss_list

# Main loop for calculations 
def progress_epoch(inputs, atom_types, data_loader, struct_labels, model, optimizer, criterion, dtype, device, non_block, valid=False, atomic_e=False):
    use_force = False
    use_stress = False
    weighted = False
    back_prop = False if valid else True
    epoch_result = logger._init_meters(struct_labels, use_force, use_stress, atomic_e)
    model.eval() if valid else model.train()

    end = time.time()
    for i, item in enumerate(data_loader):
        epoch_result['data_time'].update(time.time() - end) # save data loading time
        batch_loss, _ = loss.calculate_batch_loss(inputs, atom_types, item, model, criterion, device, non_block, epoch_result, weighted, dtype)
        if back_prop:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        epoch_result['batch_time'].update(time.time() - end) # save batch calculation time
        end = time.time()

    return epoch_result, batch_loss

class TaskGenerator():
    def __init__(self, systems_pool, num_tasks):
        self.num_tasks = num_tasks
        self.systems_pool = systems_pool

    def generate_task(self, n_support, n_query, device):
        # select self.num_tasks system files
        # system_files = ['data_HfAuO.pt', 'data_XXX.pt', ... ]
        system_files = np.random.choice(self.systems_pool, self.num_tasks,replace=False)
        task_list = []
        for system_file in system_files:
            data_generator = DataGenerator(system_file, device, n_support, n_query)
            support, query = data_generator.support_query_generate()
            task_list.append([support, query])
        return system_files, task_list

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, filename, device, n_support, n_query):
        self.filename = filename
        self.device = device
        self.n_support = n_support
        self.n_query = n_query
        self.filelist = list()

    def __len__(self):
        return len(self.filelist)

    def support_query_generate(self):
        n_support = self.n_support
        n_query = self.n_query
        tmp_data = torch.load(self.filename)
        if len(tmp_data) < n_support + n_query:
            n_support = int(len(tmp_data)*0.9)
            n_query = len(tmp_data) - n_support

        shots = np.random.choice(tmp_data, n_support+n_query, replace=False)
        support = shots[:n_support]
        query = shots[n_support:]

        return support, query

def my_collate(batch, atom_types, device, scale_factor=None, pca=None, pca_min_whiten_level=None, use_force=True, use_stress=False):
    non_blocking = True if torch.cuda.is_available() else False
    E = list()
    x = _make_empty_dict(atom_types)
    n = _make_empty_dict(atom_types)

    for item in batch:
        E.append(item['E'])
        for atype in atom_types:
            x[atype].append(torch.from_numpy(item[atype]))
            n[atype].append(item[atype].shape[0])

    E = torch.tensor(E).to(device, non_blocking=non_blocking)
    for atype in atom_types:
        x[atype] = torch.cat(x[atype], axis=0).to(device, non_blocking=non_blocking)
        n[atype] = torch.tensor(n[atype]).to(device, non_blocking=non_blocking)

    return {'x': x, 'n': n, 'E': E}

def _get_torch_device(inputs):
    if inputs['neural_network']['use_gpu'] and torch.cuda.is_available():
        cuda_num = inputs['neural_network']['GPU_number']
        device = 'cuda'+':'+str(cuda_num) if cuda_num else 'cuda'
    else:
        device = 'cpu'
    return torch.device(device)

def _make_empty_dict(atom_types):
    dic = dict()
    for atype in atom_types:
        dic[atype] = list()

    return dic
