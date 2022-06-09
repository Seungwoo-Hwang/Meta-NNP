import sys
import os
import yaml
import collections
import torch
import numpy as np
import time

default_inputs = {
    'generate_features': True,
    'preprocess': True,
    'train_model': True,
    'random_seed': None,
    #'params': dict(),
}

symmetry_function_data_default_inputs = \
        {'data':
            {
                'type'          :   'symmetry_function',
                'struct_list'   :   './structure_list',
                'refdata_format':   'vasp-out',
                'compress_outcar':  True,
                'save_directory':   './data',
                'save_list'     :   './total_list',
                'absolute_path' :   True,
                'read_force'    :   True,
                'read_stress'   :   True,
                'dx_save_sparse':   True,
            }
        }

preprocess_default_inputs = \
        {'preprocessing':
            {
                'data_list' : './total_list',
                'train_list': './train_list',
                'valid_list': './valid_list',
                'valid_rate': 0.1,
                'shuffle'   : True,
                #Scaling parameters
                'calc_scale': True,
                'scale_type': 'minmax',
                'scale_width': 1.0,
                'scale_rho' : None,
                #PCA parameters
                'calc_pca'  : True,
                'pca_whiten': True,
                'min_whiten_level': 1.0e-8,
                #Atomic weights
                'calc_atomic_weights': False,
            }
        }

model_default_inputs = \
        {'neural_network':
            {
                'train_list': './train_list',
                'valid_list': './valid_list',
                'test_list' : './test_list',
                'ref_list'  : './ref_list',

                'train'     : True,
                'test'      : False,
                'use_force' : True,
                'use_stress': True,
                'add_NNP_ref'   : False,
                'train_atomic_E': False,
                'test_atomic_E': False,
                'shuffle_dataloader': True,

                # Network related
                'nodes'     : '30-30',
                'acti_func' : 'sigmoid',
                'double_precision'  : True,
                'weight_initializer': {
                    'type'  : 'xavier normal',
                    'params': {
                        'gain'  : None,
                        'std'   : None,
                        'mean'  : None,
                        'val'   : None,
                        'sparsity':None,
                        'mode'  : None,
                        'nonlinearity': None,
                    },
                },
                'dropout'   : 0.0,
                'use_pca'   : True,
                'use_scale' : True,
                'use_atomic_weights'   : False,
                'weight_modifier': {
                    'type'  : None,
                    'params': dict(),
                },
                # Optimization
                'optimizer' : {
                    'method': 'Adam',
                    'params':
                        None
                },
                'batch_size'    : 8,
                'full_batch'    : False,
                'total_epoch'   : 1000,
                'learning_rate' : 0.0001,
                'decay_rate'    : None,
                'l2_regularization': 1.0e-6,
                # Loss function
                'loss_scale'    : 1.,
                'E_loss_type'   : 1,
                'F_loss_type'   : 1,
                'energy_coeff'  : 1.,
                'force_coeff'   : 0.1,
                'stress_coeff'  : 0.000001,
                # Logging & saving
                'show_interval' : 10,
                'save_interval':  0,
                'energy_criteria'   :   None,
                'force_criteria'    :   None,
                'stress_criteria'   :   None,
                'break_max'         :   10,
                'print_structure_rmse': False,
                'accurate_train_rmse':  True,
                # Restart
                'continue'      : None,
                'start_epoch'   : 1,
                'clear_prev_status'     : False,
                'clear_prev_optimizer'  : False,
                # Parallelism
                'use_gpu': True,
                'GPU_number'       : None,
                'subprocesses'   : 0,
                'inter_op_threads': 0,
                'intra_op_threads': 0,
            }
        }

def initialize_inputs(input_file_name, logfile):
    with open(input_file_name) as input_file:
        input_yaml = yaml.safe_load(input_file)
    if 'data' in input_yaml.keys():
        descriptor_type = input_yaml['data']['type']
    else:
        descriptor_type = 'symmetry_function'
    #params_type = input_yaml['params']

    inputs = default_inputs

    #for key in list(params_type.keys()):
    #    inputs['params'][key] = None

    data_default_inputs = get_data_default_inputs(logfile, descriptor_type=descriptor_type)
    inputs = _deep_update(inputs, data_default_inputs)
    inputs = _deep_update(inputs, preprocess_default_inputs)
    inputs = _deep_update(inputs, model_default_inputs)
    # update inputs using 'input.yaml'
    inputs = _deep_update(inputs, input_yaml, warn_new_key=True, logfile=logfile)
    #Change .T. , t to boolean
    _to_boolean(inputs)
    #add atom_types information
    #if 'atom_types' not in inputs.keys():  
    #    inputs['atom_types'] = list(params_type.keys())
    #elif not set(inputs['atom_types']) == set(params_type.keys()):
    #    inputs['atom_types'] = list(params_type.keys())
    #    logfile.write("Warning: atom_types not met with params type. Overwritting to atom_types.\n")
    #else:
    #    logfile.write("Warning: atom_types is depreciated. Use params only.\n")


    #if len(inputs['atom_types']) == 0:
    #    raise KeyError
    if not inputs['neural_network']['use_force'] and isinstance(inputs['preprocessing']['calc_atomic_weights'], dict):
        if inputs['preprocessing']['calc_atomic_weights']['type'] is not None:
            logfile.write("Warning: Force training is off but atomic weights are given. Atomic weights will be ignored.\n")
    if inputs['neural_network']['optimizer']['method'] == 'L-BFGS' and \
            not inputs['neural_network']['full_batch']:
        logfile.write("Warning: Optimization method is L-BFGS but full batch mode is off. This might results bad convergence or divergence.\n")

    if inputs['random_seed'] is None:
        inputs["random_seed"] = int(time.time()) 

    inputs['neural_network']['energy_coeff'] = float(inputs['neural_network']['energy_coeff'])
    inputs['neural_network']['force_coeff']  = float(inputs['neural_network']['force_coeff'])
    inputs['neural_network']['stress_coeff'] = float(inputs['neural_network']['stress_coeff'])

    if inputs['neural_network']['add_NNP_ref'] or inputs['neural_network']['train_atomic_E']:
        inputs['neural_network']['use_force'] = False
        inputs['neural_network']['use_stress'] = False

    return inputs

def get_data_default_inputs(logfile, descriptor_type='symmetry_function'):
    descriptor_inputs = {
        'symmetry_function': symmetry_function_data_default_inputs
    }

    if descriptor_type not in descriptor_inputs.keys():
        err = "'{}' type descriptor is not implemented.".format(descriptor_type)
        raise NotImplementedError(err)

    return descriptor_inputs[descriptor_type]

def _deep_update(source, overrides, warn_new_key=False, logfile=None, depth=0, parent='top'):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    :param dict source: base dictionary to be updated
    :param dict overrides: new dictionary
    :param bool warn_new_key: if true, warn about new keys in overrides
    :param str logfile: filename to which warnings are written (if not given warnings are written to stdout)
    :returns: updated dictionary source
    """

    if logfile is None:
        logfile = sys.stdout

    for key in overrides.keys():
        if isinstance(source, collections.Mapping):
            #if warn_new_key and depth < 2 and key not in source:
            #    logfile.write("Warning: Unidentified option in {:}: {:}\n".format(parent, key))
            if isinstance(overrides[key], collections.Mapping) and overrides[key]:
                returned = _deep_update(source.get(key, {}), overrides[key], 
                                       warn_new_key=warn_new_key, logfile=logfile,
                                       depth=depth+1, parent=key)
                source[key] = returned
            # Need list append?
            else:
                source[key] = overrides[key]
        else:
            source = {key: overrides[key]}
    return source

def _to_boolean(inputs):
    check_list =  ['generate_features', 'preprocess',  'train_model']
    data_list = ['compress_outcar','read_force','read_stress', 'dx_save_sparse', 'absolute_path']
    preprocessing_list = ['shuffle', 'calc_pca', 'pca_whiten', 'calc_scale']
    neural_network_list = ['train', 'test', 'add_NNP_ref', 'train_atomic_E', 'shuffle_dataloader', 'double_precision', 'use_force', 'use_stress',\
                        'full_batch', 'print_structure_rmse', 'accurate_train_rmse', 'use_pca', 'use_scale', 'use_atomic_weights',\
                        'clear_prev_status', 'clear_prev_optimizer', 'use_gpu']

    #True TRUE T tatrue TrUe .T. ... 
    #False FALSE F f false FaLse .F. ... 
    def convert(dic, dic_key):
        check = dic[dic_key].upper()
        if check[0] == '.':
            if check[1] == 'T':
                check = True
            elif check[1] == 'F':
                check = False
            else:
                pass
        elif check[0] == 'T':
            check = True
        elif check[0] == 'F':
            check = False
        else:
            pass
        dic[dic_key] = check

    for key in check_list:
        if not isinstance(inputs[key], bool) and isinstance(inputs[key], str):
            convert(inputs, key)

    for d_key in data_list:
        if not isinstance(inputs['data'][d_key], bool) and isinstance(inputs['data'][d_key], str):
            convert(inputs['data'], d_key)

    for p_key in preprocessing_list:
        if not isinstance(inputs['preprocessing'][p_key], bool) and isinstance(inputs['preprocessing'][p_key], str):
            convert(inputs['preprocessing'], p_key)

    for n_key in neural_network_list:
        if not isinstance(inputs['neural_network'][n_key], bool) and isinstance(inputs['neural_network'][n_key], str):
            convert(inputs['neural_network'], n_key)
