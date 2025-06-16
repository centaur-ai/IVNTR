from typing import Dict, List
import dill as pkl
import copy
import json
import os
import glob
import numpy as np
from tqdm import tqdm
from predicators.gnn.gnn_utils import GraphTransC2DDataset
from predicators.gnn.neupi_utils import two2one

def _setup_neupi_dims(dataset: GraphTransC2DDataset) -> List[int]:
    """Extract dimensions from dataset."""
    input_sample = dataset[0]['input']
    output_sample = dataset[0]['target']
    input_node_dim = input_sample['nodes'].shape[-1]
    input_edge_dim = input_sample['edges'].shape[-1]
    if 'globals' in input_sample:
        if input_sample['globals'] is not None:
            input_global_dim = input_sample['globals'].shape[-1]
        else:  
            input_global_dim = 0
    else:
        input_global_dim = 0
    output_node_dim = output_sample['nodes'].shape[-1]
    output_edge_dim = output_sample['edges'].shape[-1]
    if 'globals' in output_sample:
        if output_sample['globals'] is not None:
            output_global_dim = output_sample['globals'].shape[-1]
        else:
            output_global_dim = 0
    else:
        output_global_dim = 0
    return [
        input_node_dim, input_edge_dim, input_global_dim, output_node_dim,
        output_edge_dim, output_global_dim
    ]


def method2json(content: Dict, base_path: str):
    """
    JSON file format:
    {
        'dummy_preds': {
            pred_name: [pred_type_name]
        }
        'learned_ae_pred_info': {
            pred_temp_name:{
                idx: {
                    'quantifiers': [quantifier_str],
                    'ae_vecs': [ae_vecs], (one dim list)
                    'ent_idx': [ent_idx],

                }
            }
        }
        'node_feature_to_index': Dict[str, int],
        'edge_feature_to_index': Dict[str, int],
        'node_is_rot': Dict[str, bool],
        'edge_is_rot': Dict[str, bool],
        'input_normalizers': Dict[str, Tuple[float, float]],
        'selected_pred': {
            pred_temp_name: {
                pred_name: Tuple[str, int]
        }
    }
    """
    method_dict = {
        'pred_temp': {},
        'pred': {},
        'learned_ae_pred_info': {},
        'node_feature_to_index': {},
        'edge_feature_to_index': {},
        'node_is_rot': {},
        'edge_is_rot': {},
        'input_normalizers': {},
        'dummy_nsrts': {},
    }

    # 1. Template and info
    learned_ae_pred_info = content['learned_ae_pred_info']
    for pred_temp, learned_info in learned_ae_pred_info.items():
        pred_temp_name = pred_temp.name
        pred_temp_types = [t.name for t in pred_temp.types]
        # add its name and types to the method_dict
        assert pred_temp_name not in method_dict['pred_temp']
        method_dict['pred_temp'][pred_temp_name] = pred_temp_types
        assert pred_temp_name not in method_dict['learned_ae_pred_info']
        if 'quantifiers' in learned_info:
            method_dict['learned_ae_pred_info'][pred_temp_name] = {
                'quantifiers': copy.deepcopy(learned_info['quantifiers']),
                'model_weights': copy.deepcopy(learned_info['model_weights']),
                'ent_idx': copy.deepcopy(learned_info['ent_idx']),
            }
            method_dict['learned_ae_pred_info'][pred_temp_name]['ae_vecs'] = []
            ae_vecs = learned_info['ae_vecs']
            for pred_temp_id in range(len(ae_vecs)):
                method_dict['learned_ae_pred_info'][pred_temp_name]['ae_vecs'].append(
                    [])
                for vec in ae_vecs[pred_temp_id]:
                    one_vect = list(two2one(vec))
                    one_vect_int = [int(i) for i in one_vect]
                    method_dict['learned_ae_pred_info'][pred_temp_name]['ae_vecs'][-1].append(
                        one_vect_int
                    )
            neural_dim = _setup_neupi_dims(learned_info['example_dataset'])
            method_dict['learned_ae_pred_info'][pred_temp_name]['neural_dim'] = neural_dim
        else:
            assert len(learned_info['ae_vecs']) == 1
            assert len(learned_info['ent_idx']) == 1
            method_dict['learned_ae_pred_info'][pred_temp_name] = {
                'ent_idx': copy.deepcopy(learned_info['ent_idx'])
            }
            vec = learned_info['ae_vecs'][0]
            one_vect = list(two2one(vec))
            one_vect_int = [int(i) for i in one_vect]
            method_dict['learned_ae_pred_info'][pred_temp_name]['ae_vecs'] = [one_vect_int]
        method_dict['learned_ae_pred_info'][pred_temp_name]['learned'] = \
            learned_info['learned']
        method_dict['learned_ae_pred_info'][pred_temp_name]['provided'] = \
            learned_info['provided']

    # 2. selected_pred
    selected_pred = content['selected_pred']
    for k, v in selected_pred.items():
        pred_temp_name = k.name
        assert pred_temp_name in method_dict['pred_temp']
        assert pred_temp_name in method_dict['learned_ae_pred_info']
        method_dict['pred'][pred_temp_name] = []
        for info in v:
            pred_name = info[0].name
            pred_types = [t.name for t in info[0].types]
            method_dict['pred'][pred_temp_name].append(
                {
                    'name': pred_name,
                    'types': pred_types,
                    'i': info[1],
                    'm': info[2]
                }
            )
    
    # 3. feature stats
    node_feature_to_index = content['node_feature_to_index']
    for k, v in node_feature_to_index.items():
        method_dict['node_feature_to_index'][k] = v
    edge_feature_to_index = content['edge_feature_to_index']
    for k, v in edge_feature_to_index.items():
        method_dict['edge_feature_to_index'][k] = v
    node_is_rot = content['node_is_rot']
    method_dict['node_is_rot'] = copy.deepcopy(node_is_rot)
    edge_is_rot = content['edge_is_rot']
    method_dict['edge_is_rot'] = copy.deepcopy(edge_is_rot)
    input_normalizers = content['input_normalizers']
    if input_normalizers is None:
        method_dict['input_normalizers'] = None
    else:
        for k, v in input_normalizers.items():
            min_value_list = []
            max_value_list = []
            min_value_array = v[0]
            max_value_array = v[1]
            for i in range(len(min_value_array)):
                min_value_list.append(float(min_value_array[i]))
                max_value_list.append(float(max_value_array[i]))
            method_dict['input_normalizers'][k] = (min_value_list, max_value_list)

    # 4. NSRTs, no samplers, have not figured out how to deal with sampler...
    dummy_nsrts = content["dummy_nsrts"]
    for nsrt in dummy_nsrts:
        nsrt_name = nsrt.name
        nsrt_params = nsrt.parameters
        method_dict['dummy_nsrts'][nsrt_name] = {
            'option_name': nsrt.option.name,
            'parameters': [],
            'option_vars': [],
            'preconditions': {},
            'add_effects': {},
            'delete_effects': {},
        }
        method_dict['dummy_nsrts'][nsrt_name]['parameters'] = []
        for param in nsrt_params:
            param_name = param.name
            param_type = param.type.name
            method_dict['dummy_nsrts'][nsrt_name]['parameters'].append({
                'name': param_name,
                'type': param_type
            })
        option_name = nsrt.option.name
        method_dict['dummy_nsrts'][nsrt_name]['option_name'] = option_name
        option_vars = nsrt.option_vars
        method_dict['dummy_nsrts'][nsrt_name]['option_vars'] = []
        for var in option_vars:
            var_name = var.name
            var_type = var.type.name
            method_dict['dummy_nsrts'][nsrt_name]['option_vars'].append({
                'name': var_name,
                'type': var_type
            })
        preconditions = nsrt.preconditions
        for pre in preconditions:
            pre_name = pre.predicate.name
            # assert pre_name in method_dict['pred']
            # Tricky detail, the same predicate could appear multiple times with different variables
            curr_var = []
            if pre_name not in method_dict['dummy_nsrts'][nsrt_name]['preconditions'].keys():
                method_dict['dummy_nsrts'][nsrt_name]['preconditions'][pre_name] = []
            pre_var = pre.variables
            for v in pre_var:
                v_name = v.name
                curr_var.append(v_name)
            method_dict['dummy_nsrts'][nsrt_name]['preconditions'][pre_name].append(curr_var)
        add_effects = nsrt.add_effects
        for add in add_effects:
            add_name = add.predicate.name
            # assert add_name in method_dict['pred']
            # Tricky detail, the same predicate could appear multiple times with different variables
            curr_var = []
            if add_name not in method_dict['dummy_nsrts'][nsrt_name]['add_effects'].keys():
                method_dict['dummy_nsrts'][nsrt_name]['add_effects'][add_name] = []
            add_var = add.variables
            for v in add_var:
                v_name = v.name
                curr_var.append(v_name)
            method_dict['dummy_nsrts'][nsrt_name]['add_effects'][add_name].append(curr_var)
        delete_effects = nsrt.delete_effects
        for delete in delete_effects:
            delete_name = delete.predicate.name
            # Tricky detail, the same predicate could appear multiple times with different variables
            curr_var = []
            # assert delete_name in method_dict['pred']
            if delete_name not in method_dict['dummy_nsrts'][nsrt_name]['delete_effects'].keys():
                method_dict['dummy_nsrts'][nsrt_name]['delete_effects'][delete_name] = []
            delete_var = delete.variables
            for v in delete_var:
                v_name = v.name
                curr_var.append(v_name)
            method_dict['dummy_nsrts'][nsrt_name]['delete_effects'][delete_name].append(curr_var)

    with open(base_path, 'w') as f:
        json.dump(method_dict, f, indent=4)

for seed in [0,1,2,3,4]:
    base_path = f'saved_approaches/final/view_plan_trivial/biplan_{seed}'
    all_files = glob.glob(os.path.join(base_path, '*.neupi_info'))
    for file in tqdm(all_files):
        name = os.path.basename(file)
        name = name.replace('.neupi_info', '')
        with open(file, 'rb') as f:
            content = pkl.load(f)
        method2json(content, os.path.join(base_path, f'{name}.json'))
