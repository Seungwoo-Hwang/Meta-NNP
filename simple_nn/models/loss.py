import torch
from ase import units


def calculate_batch_loss(inputs, atom_types, item, model, criterion, device, non_block, epoch_result, weighted, dtype):
    n_batch = item['E'].size(0)
    calc_results = dict()
    x, atomic_E, E_, n_atoms = calculate_E(atom_types, item, model, device, non_block)
    e_loss = get_e_loss(atom_types, inputs['neural_network']['E_loss_type'], atomic_E, E_, n_atoms, item, criterion,\
                    epoch_result, dtype, device, non_block, n_batch)
    batch_loss = e_loss
    calc_results['E'] = E_
    epoch_result['losses'].update(batch_loss.detach().item(), n_batch)

    return batch_loss, calc_results

def calculate_E(atom_types, item, model, device, non_block):
    x = dict()
    atomic_E = dict()
    E_ = 0
    n_atoms = 0
    for atype in atom_types:
        x[atype] = item['x'][atype].to(device=device, non_blocking=non_block).requires_grad_(True)
        atomic_E[atype] = None
        if x[atype].size(0) != 0:
            atomic_E[atype] = model.nets[atype](x[atype])
            atomic_E[atype] = model.nets[atype](x[atype]).reshape(item['n'][atype].size(0), -1)
            E_ += torch.sum(atomic_E[atype], axis=1)
        n_atoms += item['n'][atype].to(device=device, non_blocking=non_block)

    return x, atomic_E, E_, n_atoms

def get_e_loss(atom_types, loss_type, atomic_E, E_, n_atoms, item, criterion, progress_dict, dtype, device, non_block, n_batch):
    if loss_type == 1:
        e_loss = criterion(E_.squeeze() / n_atoms, item['E'].type(dtype).to(device=device, non_blocking=non_block) / n_atoms)
    elif loss_type == 2:
        e_loss = criterion(E_.squeeze() / n_atoms, item['E'].type(dtype).to(device=device, non_blocking=non_block) / n_atoms) * n_atoms
    else:
        e_loss = criterion(E_.squeeze(), item['E'].type(dtype).to(device=device, non_blocking=non_block))

    w_e_loss = torch.mean(e_loss)
    for i in range(len(e_loss)):
        progress_dict['e_err']['None'].update(e_loss[i].detach().item())
    #progress_dict['tot_e_err'].update(torch.mean(e_loss).detach(), n_batch)

    return w_e_loss
