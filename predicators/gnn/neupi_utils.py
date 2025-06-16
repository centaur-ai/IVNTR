from __future__ import division

import re
import copy
import collections
import logging
import time
import wandb
import itertools
from tqdm import tqdm
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F
from predicators.settings import CFG

from .gnn_utils import discrete_graph, _create_super_graph
from predicators.structs import MDPDataset, Type, DummyPredicate, LowLevelTrajectory, \
    State, GroundAtom, _Option, Predicate, Dataset
from predicators.utils import abstract

def two2one(two: torch.Tensor) -> torch.Tensor:
    """Convert a 2(1)D ae vector to a 1D tensor."""
    if two.shape[1] == 2:
        one = two[:, 0] * 1 + two[:, 1] * 2
    elif two.shape[1] == 1:
        one = two[:, 0]
    return one

def one2two(one: torch.Tensor, channels: int) -> torch.Tensor:
    """Convert a 1D ae vector to a 2D tensor."""
    cate_vector_sampled = torch.tensor(one).unsqueeze(0)  
    # from cate to n_row x n_channel
    vector_sampled = torch.zeros((1, one.shape[0], channels), dtype=int)
    if channels == 1:
        vector_sampled[:, :, 0] = cate_vector_sampled
    else:
        assert channels == 2, "Only support 1 or 2 channels"
        possible_values = torch.tensor([[0, 0], [1, 0], [0, 1]])
        vector_sampled = possible_values[cate_vector_sampled]
    two = vector_sampled.squeeze(0)
    return two

def prob_three2two(three: torch.Tensor) -> torch.Tensor:
    one = three.argmax(dim=1)
    two = one2two(one, 2)
    return two

def gen_pretty_pred_name(basic_pred_name: str,
                         type_comb: List[Type],
                         quantifier: str,
                         w_negation: bool) -> str:
    # Create variable placeholders for type_comb, e.g., ?sat, ?obj
    quantified_placeholders = []
    for i, t in enumerate(type_comb):
        # Create variable placeholder starting with '?'
        quantified_placeholders.append(f"?{t[:3].lower()}")  # Use the first 3 letters of the type name in lowercase

    # Generate the final string
    if w_negation:
        result_string = f"{quantifier}:{''.join(quantified_placeholders)}, Not-{basic_pred_name}"
    else:
        result_string = f"{quantifier}:{''.join(quantified_placeholders)}, -{basic_pred_name}"
    return result_string

def name2quantifier_types(quantifier_str: str, all_types: List[str]) -> Tuple[str, List[str]]:
    """Extract the quantifier and types from the given name."""
    all_types_short = {t[:3].lower(): t for t in all_types}
    # Split the name by ':'
    parts = quantifier_str.split(':')
    # Extract the quantifier
    quantifier = parts[0]
    # Extract the types
    types = parts[1].split('?')
    # Check if the types are valid
    tp = []
    for t in types:
        if t == '':
            continue
        assert t in all_types_short.keys(), f"Invalid type {t}"
        tp.append(all_types_short[t])
    return quantifier, tp

def parse_basic_pred_name(basic_pred_name: str, \
                          original_type_names: List[str]) -> Tuple[List[int], List[int]]:
    # Regex to match the pattern ?[type] where [type] is implicit
    pattern = r"\?(\w+\d*)"
    
    # Find all matches in the string
    matches = re.findall(pattern, basic_pred_name)
    
    # Lists to store the required information
    empty_positions = []
    number_list = []
    
    # Iterate over the matches and their indices
    for i, match in enumerate(matches):
        ori_type_name = original_type_names[i][:3].lower()
        if match == ori_type_name:  # If there's no number after ?type
            empty_positions.append(i)
        else:
            assert ori_type_name in match, f"Type mismatch: {ori_type_name} vs {match}"
            # Extract the number from the match
            number_list.append(int(match[len(ori_type_name):]))
    
    return empty_positions, number_list

def mdp2positive(mdpdataset: MDPDataset) -> Dataset:
    """Convert an MDPDataset to a Dataset by removing the negative examples and reward."""
    trajs = []
    for mdp_traj in mdpdataset.trajectories:
        if mdp_traj.is_pos:
            states = mdp_traj.states
            actions = mdp_traj.actions
            is_demo = mdp_traj.is_demo
            train_task_id = mdp_traj.train_task_idx
            traj = LowLevelTrajectory(states, actions, is_demo, train_task_id)
            trajs.append(traj)
    dataset = Dataset(trajs)
    return dataset

# Function to generate all possible combinations of k columns and return the selected matrix with indices
def select_columns(matrix, provided_matrix, k):
    n_cols = matrix.shape[1]
    # Generate all combinations of selecting k columns
    col_indices_combinations = combinations(range(n_cols), k)
    
    for combo in col_indices_combinations:
        # Yield the matrix formed by selecting k columns and the column indices
        combined_matrix = torch.cat([provided_matrix, matrix[:, combo]], dim=1)
        yield combined_matrix, combo

def check_neupi_all_ae_vectors(model_paths: List[str]) -> None:
    """Check if all the autoencoder models have the same number of vectors."""
    ae_vectors = []
    ae_guidance = []
    for model_path in tqdm(model_paths):
        vec_path = model_path.replace('model', 'ae_vector')
        guidance_path = model_path.replace('model', 'guidance')
        ae_vec = torch.load(vec_path)
        ae_guid = torch.load(guidance_path)
        ae_vectors.append(ae_vec.clone())
        ae_guidance.append(ae_guid.clone())
    tgt_vect = torch.zeros_like(ae_vectors[0])
    tgt_vect[1, 1] = 1
    tgt_vect[2, 0] = 1
    for i in range(len(ae_vectors)):
        if torch.equal(ae_vectors[i], tgt_vect):
            logging.info(f"Model {i} has the same vector as the target vector:")
            logging.info(f"Add: {ae_vectors[i][:,0]}")
            logging.info(f"Delete: {ae_vectors[i][:,1]}")
            logging.info(f"Guidance: {ae_guidance[i]}")
    return

def get_single_neupi_prediction(model: Any,
                                single_input: Dict,
                                temperature: float,
                                boundary: float,
                                device: Optional[torch.device] = None) -> Dict:
    """Get a prediction from the given model on the given input."""
    model.to(device)
    model.train(False)
    model.eval()
    graph = {}
    inputs = _create_super_graph([single_input], device=device)
    output_logits = model(inputs.copy())[-1]
    output_binary = discrete_graph(output_logits, temperature=temperature, boundary=boundary)
    graph['nodes'] = output_binary['nodes'].cpu().detach().numpy()
    graph_edges = torch.zeros((output_binary['n_node'][0,0], output_binary['n_node'][0,0], \
                            output_binary['edges'].shape[1])).to(output_binary['edges'].device)
    graph_edges[output_binary['senders'], output_binary['receivers'], :] \
        = output_binary['edges']
    graph['edges'] = graph_edges.cpu().detach().numpy()
    if output_binary['globals'] is not None:
        graph['globals'] = output_binary['globals'].cpu().detach().numpy()
    return graph

def js_divergence_sigmoid(p, q, sigmoid=True):
    """
    Compute the Jensen-Shannon divergence between two distributions using sigmoid.
    :param p: Tensor, shape (B, 1) representing the logits of distribution p.
    :param q: Tensor, shape (B, 1) representing the logits of distribution q.
    :return: Tensor, representing the JS divergence for each pair in the batch.
    """
    # Apply sigmoid to logits to get probability distributions
    if sigmoid:
        p = torch.sigmoid(p)
        q = torch.sigmoid(q)
    
    # Clamp the probabilities to avoid log(0)
    p = torch.clamp(p, min=1e-10, max=1-1e-10)
    q = torch.clamp(q, min=1e-10, max=1-1e-10)
    # Calculate the midpoint distribution
    m = 0.5 * (p + q)
    
    # Compute the KL divergence between p and m, and between q and m
    kl_pm = F.kl_div(m.log(), p, reduction='batchmean')
    kl_qm = F.kl_div(m.log(), q, reduction='batchmean')
    
    # JS Divergence is the average of the two KL divergences
    js = 0.5 * (kl_pm + kl_qm)
    
    return js

def neupi_supervise_criterion(
                    outputs_logits: Tuple[torch.Tensor],
                    targets: Tuple[torch.Tensor],
                    action_info: Dict,
                    super_label: Dict,
                    device: Optional[torch.device] = None,
                    debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    # Supervise the unChanged Predicates
    # output_binary_s, output_binary_s_ = outputs_binary
    bce_loss = torch.nn.BCEWithLogitsLoss()
    outputs_logits_s, outputs_logits_s_ = outputs_logits
    output_target, output_target_ = targets

    # Supervise the unChanged Unary Predicates
    nodes_s = outputs_logits_s["nodes"]
    nodes_s_ = outputs_logits_s_["nodes"]
    nodes_target = output_target["nodes"]
    nodes_target_ = output_target_["nodes"]
    if not (nodes_target == super_label['ignore']).all():
        assert (output_target["edges"]== super_label['ignore']).all(), \
                "All edges Should be ignored"
        _, node_indexs = torch.where(nodes_target !=super_label['ignore'])
        assert (node_indexs == node_indexs[0]).all(), "All nodes should be the same index"
        node_index = node_indexs[0]
        nodes_target = nodes_target[:, node_index]
        nodes_target_ = nodes_target_[:, node_index]
        # Supervise the unChanged Unary Predicates
        # First type of non-change
        node_mask_s = nodes_target == super_label['non_change_1']
        node_mask_s_ = nodes_target_ == super_label['non_change_1']
        assert torch.equal(node_mask_s, node_mask_s_), "Mask should be the same for s and s'"
        # Apply the mask to the nodes and target nodes
        masked_nodes_s = nodes_s[node_mask_s]
        masked_nodes_s_ = nodes_s_[node_mask_s_]
            
        if masked_nodes_s.shape[0] == 0:
            node_loss_non_change = torch.tensor(0.0).to(device)
        else:
            # the two should be the same, normalze loss scale
            if debug:
                node_loss_non_change = js_divergence_sigmoid(masked_nodes_s, \
                                                                masked_nodes_s_, sigmoid = False)
            else:
                node_loss_non_change = js_divergence_sigmoid(masked_nodes_s, masked_nodes_s_)

        # Second type of non-change
        node_mask_s = nodes_target ==super_label['non_change_2']
        node_mask_s_ = nodes_target_ ==super_label['non_change_2']
        assert torch.equal(node_mask_s, node_mask_s_), "Mask should be the same for s and s'"
        # Apply the mask to the nodes and target nodes
        masked_nodes_s = nodes_s[node_mask_s]
        masked_nodes_s_ = nodes_s_[node_mask_s_]
            
        if masked_nodes_s.shape[0] == 0:
            node_loss_non_change += torch.tensor(0.0).to(device)
        else:
            # the two should be the same, normalze loss scale
            if debug:
                node_loss_non_change += js_divergence_sigmoid(masked_nodes_s, \
                                                                masked_nodes_s_, sigmoid = False)
            else:
                node_loss_non_change += js_divergence_sigmoid(masked_nodes_s, masked_nodes_s_)

        # Third type of non-change
        node_mask_s = nodes_target ==super_label['non_change_3']
        node_mask_s_ = nodes_target_ ==super_label['non_change_3']
        assert torch.equal(node_mask_s, node_mask_s_), "Mask should be the same for s and s'"
        # Apply the mask to the nodes and target nodes
        masked_nodes_s = nodes_s[node_mask_s]
        masked_nodes_s_ = nodes_s_[node_mask_s_]
            
        if masked_nodes_s.shape[0] == 0:
            node_loss_non_change += torch.tensor(0.0).to(device)
        else:
            # the two should be the same, normalze loss scale
            if debug:
                node_loss_non_change += js_divergence_sigmoid(masked_nodes_s, \
                                                                masked_nodes_s_, sigmoid = False)
            else:
                node_loss_non_change += js_divergence_sigmoid(masked_nodes_s, masked_nodes_s_)

        # Supervise the Changed Unary Predicates
        # BCE loss for nodes
        all_nodes = torch.cat([nodes_s, nodes_s_], dim=0).flatten()
        all_tgt_nodes = torch.cat([nodes_target, nodes_target_], dim=0).flatten()
        # 0/1 means supervision signal
        node_mask = (all_tgt_nodes ==super_label['change_pos']) | \
                    (all_tgt_nodes ==super_label['change_neg'])
        # Apply the mask to the nodes and target nodes
        masked_nodes = all_nodes[node_mask]
        masked_tgt_nodes = all_tgt_nodes[node_mask]
        # weighted BCE loss
        pos_mask = masked_tgt_nodes ==super_label['change_pos']
        neg_mask = masked_tgt_nodes ==super_label['change_neg']

        if pos_mask.sum() < neg_mask.sum():
            # Sample negative samples
            neg_indices = torch.where(neg_mask)[0]
            sampled_neg_indices = neg_indices[torch.randperm(neg_indices.shape[0])[:pos_mask.sum().item()]]
            masked_nodes = torch.cat([masked_nodes[pos_mask], masked_nodes[sampled_neg_indices]], dim=0)
            masked_tgt_nodes = torch.cat([masked_tgt_nodes[pos_mask], masked_tgt_nodes[sampled_neg_indices]], dim=0)

        if masked_nodes.shape[0] == 0:
            node_loss_change = torch.tensor(0.0).to(device)
        else:
            if debug:
                masked_nodes_logits = torch.log(masked_nodes / (1 - masked_nodes + 1e-10) + 1e-10)
                node_loss_change = bce_loss(masked_nodes_logits, masked_tgt_nodes)
            else:
                node_loss_change = bce_loss(masked_nodes, masked_tgt_nodes)

        return node_loss_non_change, node_loss_change
    
    else:
        original_edge_tgt = output_target["edges"]
        assert not (original_edge_tgt ==super_label['ignore']).all(), \
                "Edges should be all ignored"
        _, edge_idexs = torch.where(original_edge_tgt !=super_label['ignore'])
        assert (edge_idexs == edge_idexs[0]).all(), "All edges should be the same index"
        edge_index = edge_idexs[0]
        # Supervise the unChanged Binary Predicates
        output_s_edges = torch.zeros([output_target["nodes"].shape[0], 
                                        output_target["nodes"].shape[0],
                                        1]).to(output_target["edges"].device)
        output_s__edges = torch.zeros([output_target_["nodes"].shape[0],
                                        output_target_["nodes"].shape[0],
                                        1]).to(output_target_["edges"].device)

        # get the predicted edges for the provided predicates, use binary here
        output_s_edges[outputs_logits_s["senders"], outputs_logits_s["receivers"]] \
                = outputs_logits_s["edges"]
        output_s__edges[outputs_logits_s_["senders"], outputs_logits_s_["receivers"]] \
                = outputs_logits_s_["edges"]

        # format target edges
        output_target_edges = torch.zeros_like(output_s_edges)
        output_target__edges = torch.zeros_like(output_s__edges)
        output_target_edges[output_target["senders"], output_target["receivers"]] \
                = output_target["edges"][:, edge_index:edge_index+1]
        output_target__edges[output_target_["senders"], output_target_["receivers"]] \
                = output_target_["edges"][:, edge_index:edge_index+1]

        # First type of non-change
        edge_mask_s_1 = output_target_edges ==super_label['non_change_1']
        edge_mask_s_2 = output_target_edges ==super_label['non_change_2']
        edge_mask_s_3 = output_target_edges ==super_label['non_change_3']
        # edge_mask_s_ = output_target__edges < 0
        # assert torch.equal(edge_mask_s, edge_mask_s_), "Mask should be the same for s and s'"
        # Note, this is very important, we should mask the edges
        # zero mask is shared in one batch, across different edge matrixes
        # Create zero mask
        num_nodes = output_target["nodes"].shape[0]
        zero_mask = torch.zeros([num_nodes, num_nodes, 1], dtype=torch.bool, device=device)

        # Indices for the batch nodes
        node_indices = torch.cumsum(torch.cat([torch.tensor([0], device=device), output_target["n_node"].flatten()]), dim=0)

        # Use broadcasting to set the subgraph connections
        for start, end in zip(node_indices[:-1], node_indices[1:]):
            zero_mask[start:end, start:end, :] = True

        # Remove self connections by creating a diagonal mask and using broadcasting
        diag_mask = torch.ones([num_nodes, num_nodes], dtype=torch.bool, device=device)
        diag_mask.fill_diagonal_(False)
        zero_mask = zero_mask & diag_mask.unsqueeze(-1)
        # Apply masks 1
        edge_mask_s_1 = edge_mask_s_1 & zero_mask
        edge_mask_s_2 = edge_mask_s_2 & zero_mask
        edge_mask_s_3 = edge_mask_s_3 & zero_mask

        # Mask 1 edges
        masked_edges_s = output_s_edges[edge_mask_s_1]
        masked_edges_s_ = output_s__edges[edge_mask_s_1]
        if masked_edges_s.shape[0] == 0:
            edge_loss_non_change = torch.tensor(0.0).to(device)
        else:
            if debug:
                edge_loss_non_change = js_divergence_sigmoid(masked_edges_s, \
                                                                masked_edges_s_, sigmoid = False)
            else:
                edge_loss_non_change = js_divergence_sigmoid(masked_edges_s, masked_edges_s_)

        # Mask 2 edges
        masked_edges_s = output_s_edges[edge_mask_s_2]
        masked_edges_s_ = output_s__edges[edge_mask_s_2]
        if masked_edges_s.shape[0] == 0:
            edge_loss_non_change += torch.tensor(0.0).to(device)
        else:
            if debug:
                edge_loss_non_change += js_divergence_sigmoid(masked_edges_s, \
                                                                masked_edges_s_, sigmoid = False)
            else:
                edge_loss_non_change += js_divergence_sigmoid(masked_edges_s, masked_edges_s_)

        # Mask 3 edges
        masked_edges_s = output_s_edges[edge_mask_s_3]
        masked_edges_s_ = output_s__edges[edge_mask_s_3]
        if masked_edges_s.shape[0] == 0:
            edge_loss_non_change += torch.tensor(0.0).to(device)
        else:
            if debug:
                edge_loss_non_change += js_divergence_sigmoid(masked_edges_s, \
                                                                masked_edges_s_, sigmoid = False)
            else:
                edge_loss_non_change += js_divergence_sigmoid(masked_edges_s, masked_edges_s_)

        # Supervise the Changed Unary Predicates
        # format output edges
        output_s_edges = torch.zeros([num_nodes, 
                                        num_nodes,
                                        1]).to(output_target["edges"].device)
        output_s__edges = torch.zeros([num_nodes,
                                        num_nodes,
                                        1]).to(output_target_["edges"].device)
        # get the predicted edges for the provided predicates, use logits here
        output_s_edges[outputs_logits_s["senders"], outputs_logits_s["receivers"]] \
            = outputs_logits_s["edges"]
        output_s__edges[outputs_logits_s_["senders"], outputs_logits_s_["receivers"]] \
            = outputs_logits_s_["edges"]
        # BCE loss for edges
        all_edges = torch.cat([output_s_edges.flatten(), output_s__edges.flatten()], dim=0)
        all_tgt_edges = torch.cat([output_target_edges.flatten(), output_target__edges.flatten()], dim=0)
        # mask the edges, # 0/1 means supervision signal, do not supervise
        edge_mask = (output_target_edges ==super_label['change_pos']) | \
                    (output_target_edges ==super_label['change_neg'])
        edge_mask = edge_mask & zero_mask
        edge_mask = torch.cat([edge_mask.flatten()]*2, dim=0)
        masked_edges = all_edges[edge_mask]
        masked_tgt_edges = all_tgt_edges[edge_mask]
        # balance the pos and neg samples
        pos_mask = masked_tgt_edges ==super_label['change_pos']
        neg_mask = masked_tgt_edges ==super_label['change_neg']
        if pos_mask.sum() < neg_mask.sum():
            # Sample negative samples
            neg_indices = torch.where(neg_mask)[0]
            sampled_neg_indices = neg_indices[torch.randperm(neg_indices.shape[0])[:pos_mask.sum().item()]]
            masked_edges = torch.cat([masked_edges[pos_mask], masked_edges[sampled_neg_indices]], dim=0)
            masked_tgt_edges = torch.cat([masked_tgt_edges[pos_mask], masked_tgt_edges[sampled_neg_indices]], dim=0)

        if masked_edges.shape[0] == 0:
            edge_loss_change = torch.tensor(0.0).to(device)
        else:
            if debug:
                masked_edges_logits = torch.log(masked_edges / (1 - masked_edges + 1e-10) + 1e-10)
                edge_loss_change = bce_loss(masked_edges_logits, masked_tgt_edges)
            else:
                edge_loss_change = bce_loss(masked_edges, masked_tgt_edges)

        return edge_loss_non_change, edge_loss_change
        
def train_predicate_model(
    iteration: int,
    model: Any,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    super_label: Dict,
    num_epochs: int,
    gumbel_temperature: float,
    val_freq: int = 1,
    device: Optional[torch.device] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    quick_skip: Optional[Dict] = None,
) -> OrderedDict[str, torch.Tensor]:
    """Optimize the model and save checkpoints."""
    since = time.perf_counter()

    # Note: best_seen_model_weights is measured on validation (not train) loss.
    best_seen_model_weights: OrderedDict[str, torch.Tensor] = collections.OrderedDict({})
    # # debug, no actual training
    # return model.state_dict()
    best_seen_model_train_loss = np.inf
    best_seen_running_validation_loss = np.inf

    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            logging.info(f'Iteration {iteration} Epoch {epoch}/{num_epochs - 1}')
            logging.info('-' * 10)

        running_loss = {
            'non_change_train': [],
            'change_train': [],
            'non_change_val': [],
            'change_val': [],
            'train_loss': [],
            'val_loss': [],
        }
        model.train(True)
        model.to(device)
        total_data_time = 0.0
        total_model_time = 0.0

        # Iterate over data.
        for idx, data in enumerate(train_dataloader):
            # this is a big graph built from all the batches
            # transition pair1
            input = data['input']
            input_ = data['input_']
            target = data['target']
            target_ = data['target_']
            action_info = data['action_info']

            # End data loading time
            if idx > 0:
                data_time = time.time() - start_data_time
                total_data_time += data_time

            # zero the parameter gradients
            optimizer.zero_grad()

            # put input dict on device and start measuring model processing time
            start_model_time = time.time()
            
            if device is not None:
                for key, val in input.items():
                    input[key] = val.to(device) if val is not None else val
                for key, val in input_.items():
                    input_[key] = val.to(device) if val is not None else val
                for key, val in target.items():
                    target[key] = val.to(device) if val is not None else val
                for key, val in target_.items():
                    target_[key] = val.to(device) if val is not None else val
            
            # the model partition the continous input graph into discrete graphs (groundings of predicates)
            logits_output = model(input.copy())[-1]
            logits_output_ = model(input_.copy())[-1]
            outputs_logits = (logits_output, logits_output_)
            targets = (target, target_)

            loss = torch.tensor(0.0).to(device)
            if device is not None:
                loss = loss.to(device)

            # loss is not averaged over the batch (single sample scale)
            # aet loss accepts gumbel-sigmoid output
            non_change_loss, change_loss = neupi_supervise_criterion(
                outputs_logits, targets, action_info, super_label, device=device)
            running_loss['non_change_train'].append(non_change_loss.item()) # aet_loss is already normalized
            loss += non_change_loss
            running_loss['change_train'].append(change_loss.item())
            loss += change_loss

            # backward + optimize
            loss.backward()
            optimizer.step()
            # End model processing time
            model_time = time.time() - start_model_time
            total_model_time += model_time
            # statistics
            running_loss['train_loss'].append(loss.item())

            # Log per-sample times
            # if idx > 0:
            #     logging.info(f"Epoch {epoch}, Sample {idx}, data loading time: {data_time:.4f} seconds")
            #     logging.info(f"Epoch {epoch}, Sample {idx}, model processing time: {model_time:.4f} seconds")
            # logging.info(f"Epoch {epoch}, Sample {idx}, non_change_loss: {non_change_loss.item()}, change_loss: {change_loss.item()}")
            if wandb.run is not None:
                wandb.log({"non_change_loss_single": non_change_loss.item(), "change_loss_single": change_loss.item(), "train_loss_single": loss.item()})

            start_data_time = time.time()

        # step the scheduler
        if scheduler is not None:
            scheduler.step()
        logging.info(f"Epoch {epoch}, model processing time: {total_model_time:.4f} seconds")
        logging.info(f"Epoch {epoch}, data loading time: {total_data_time:.4f} seconds")
        logging.info(f"Epoch {epoch}, average training loss (change): {np.mean(running_loss['change_train'])}")
        logging.info(f"Epoch {epoch}, average training loss (non-change): {np.mean(running_loss['non_change_train'])}")
        # Calculate average data and model processing times per sample
        if (epoch + 1) % val_freq == 0:
            model.train(False)
            model.to(device)
            for idx, data in enumerate(val_dataloader):
                # this is a big graph built from all the batches
                # transition pair1
                input = data['input']
                input_ = data['input_']
                target = data['target']
                target_ = data['target_']
                action_info = data['action_info']

                # End data loading time
                # if idx > 0:
                #     data_time = time.time() - start_data_time
                #     total_data_time += data_time

                # zero the parameter gradients
                optimizer.zero_grad()

                # put input dict on device and start measuring model processing time
                start_model_time = time.time()
                
                if device is not None:
                    for key, val in input.items():
                        input[key] = val.to(device) if val is not None else val
                    for key, val in input_.items():
                        input_[key] = val.to(device) if val is not None else val
                    for key, val in target.items():
                        target[key] = val.to(device) if val is not None else val
                    for key, val in target_.items():
                        target_[key] = val.to(device) if val is not None else val
                
                # the model partition the continous input graph into discrete graphs (groundings of predicates)
                logits_output = model(input.copy())[-1]
                logits_output_ = model(input_.copy())[-1]
                outputs_logits = (logits_output, logits_output_)
                targets = (target, target_)

                loss = torch.tensor(0.0).to(device)
                if device is not None:
                    loss = loss.to(device)

                # loss is not averaged over the batch (single sample scale)
                # aet loss accepts gumbel-sigmoid output
                non_change_loss, change_loss = neupi_supervise_criterion(
                    outputs_logits, targets, action_info, super_label, device=device)
                running_loss['non_change_val'].append(non_change_loss.item())
                loss += non_change_loss
                running_loss['change_val'].append(change_loss.item())
                loss += change_loss

                # statistics
                running_loss['val_loss'].append(loss.item())                

        avg_data_time = total_data_time / (idx+1)
        avg_model_time = total_model_time / (idx+1)
        running_loss = {k: np.mean(v) if len(v) > 0 else 100.0 for k, v in running_loss.items()}
        if wandb.run is not None:
            wandb.log({"non_change_train": running_loss['non_change_train'], "change_train": running_loss['change_train'],
                        "train_loss_avg": running_loss['train_loss'], "non_change_val": running_loss['non_change_val'],
                            "change_val": running_loss['change_val'], "val_loss_avg": running_loss['val_loss']})

        total_loss_val = running_loss['val_loss']
        total_loss_train = running_loss['train_loss']
        # logging.info(f"Epoch {epoch}, train loss: {total_loss_train}")
        if total_loss_val < best_seen_running_validation_loss:
            best_seen_running_validation_loss = total_loss_val
            best_seen_model_weights = model.state_dict()
            logging.info(
                f"Found new best model with val loss {best_seen_running_validation_loss}"
                f"at epoch {epoch}")
        if (quick_skip is not None) and ((epoch + 1) in quick_skip):
            desired_loss = quick_skip[(epoch + 1)]
            if best_seen_running_validation_loss > desired_loss:
                logging.info(f"Skip the rest of the training at epoch {epoch}")
                break
        total_loss_train = running_loss['train_loss']
        if total_loss_train < best_seen_model_train_loss:
            best_seen_model_train_loss = total_loss_train

    time_elapsed = time.perf_counter() - since
    num_min = time_elapsed // 60
    num_sec = time_elapsed % 60

    # if not do_validation:
    #     train_loss = running_loss['train']
    #     logging.info(f"Training complete in {num_min:.0f}m {num_sec:.0f}s "
    #                  f"with train loss {train_loss}")
    #     return model.state_dict()
    logging.info(
        f"Training complete in {num_min:.0f}m {num_sec:.0f}s "
        f"with train loss {best_seen_model_train_loss:.5f} and validation "
        f"loss {best_seen_running_validation_loss:.5f}")

    assert best_seen_model_weights
    return best_seen_model_weights, best_seen_running_validation_loss

def train_init_dummy_model(
    model: Any,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[Tuple[Dict], Tuple[Dict], Tuple[Dict]], torch.Tensor],
    num_epochs: int,
    gumbel_temperature: float,
    val_freq: int = 1,
    device: Optional[torch.device] = None,
) -> OrderedDict[str, torch.Tensor]:
    """Optimize the model and save checkpoints."""
    since = time.perf_counter()

    # Note: best_seen_model_weights is measured on validation (not train) loss.
    best_seen_model_weights: OrderedDict[str, torch.Tensor] = collections.OrderedDict({})
    best_seen_model_train_loss = np.inf
    best_seen_running_validation_loss = np.inf

    for epoch in range(num_epochs):
        if epoch % 2 == 0:
            logging.info(f'Epoch {epoch}/{num_epochs - 1}')
            logging.info('-' * 10)

        running_loss = {
            'non_change_train': [],
            'change_train': [],
            'non_change_val': [],
            'change_val': [],
            'train_loss': [],
            'val_loss': [],
        }
        model.train(True)
        model.to(device)
        total_data_time = 0.0
        total_model_time = 0.0

        # Iterate over data.
        for idx, data in enumerate(train_dataloader):
            # this is a big graph built from all the batches
            # transition pair1
            input = data['input']
            input_ = data['input_']
            target = data['target']
            target_ = data['target_']
            action_info = data['action_info']

            # End data loading time
            # if idx > 0:
            #     data_time = time.time() - start_data_time
            #     total_data_time += data_time

            # zero the parameter gradients
            optimizer.zero_grad()

            # put input dict on device and start measuring model processing time
            start_model_time = time.time()
            
            if device is not None:
                for key, val in input.items():
                    input[key] = val.to(device) if val is not None else val
                for key, val in input_.items():
                    input_[key] = val.to(device) if val is not None else val
                for key, val in target.items():
                    target[key] = val.to(device) if val is not None else val
                for key, val in target_.items():
                    target_[key] = val.to(device) if val is not None else val
            
            # the model partition the continous input graph into discrete graphs (groundings of predicates)
            logits_output = model(input.copy())
            logits_output_ = model(input_.copy())
            # discretize the output graphs
            # binary_output = logits_output
            # binary_output_ = logits_output_
            
            # outputs_binary = (binary_output, binary_output_)
            outputs_logits = (logits_output, logits_output_)
            targets = (target, target_)

            loss = torch.tensor(0.0).to(device)
            if device is not None:
                loss = loss.to(device)

            # loss is not averaged over the batch (single sample scale)
            # aet loss accepts gumbel-sigmoid output
            non_change_loss, change_loss = criterion(outputs_logits, targets, action_info)
            running_loss['non_change_train'].append(non_change_loss.item()) # aet_loss is already normalized
            loss += non_change_loss
            running_loss['change_train'].append(change_loss.item())
            loss += change_loss

            # backward + optimize
            loss.backward()
            optimizer.step()

            # End model processing time
            model_time = time.time() - start_model_time
            total_model_time += model_time
            # statistics
            running_loss['train_loss'].append(loss.item())

            # Log per-sample times
            # if idx > 0:
            #     logging.info(f"Epoch {epoch}, Sample {idx}, data loading time: {data_time:.4f} seconds")
            #     logging.info(f"Epoch {epoch}, Sample {idx}, model processing time: {model_time:.4f} seconds")
            logging.info(f"Epoch {epoch}, Sample {idx}, non_change_loss: {non_change_loss.item()}, change_loss: {change_loss.item()}")
            # wandb.log({"aet_loss_single": aet_loss.item(), "superv_loss_single": superv_loss.item(), "train_loss_single": loss.item()})

            start_data_time = time.time()

        # Calculate average data and model processing times per sample
        if epoch % val_freq == 0:
            model.train(False)
            model.to(device)
            for idx, data in enumerate(val_dataloader):
                # this is a big graph built from all the batches
                # transition pair1
                input = data['input']
                input_ = data['input_']
                target = data['target']
                target_ = data['target_']
                action_info = data['action_info']

                # End data loading time
                if idx > 0:
                    data_time = time.time() - start_data_time
                    total_data_time += data_time

                # zero the parameter gradients
                optimizer.zero_grad()

                # put input dict on device and start measuring model processing time
                start_model_time = time.time()
                
                if device is not None:
                    for key, val in input.items():
                        input[key] = val.to(device) if val is not None else val
                    for key, val in input_.items():
                        input_[key] = val.to(device) if val is not None else val
                    for key, val in target.items():
                        target[key] = val.to(device) if val is not None else val
                    for key, val in target_.items():
                        target_[key] = val.to(device) if val is not None else val
                
                # the model partition the continous input graph into discrete graphs (groundings of predicates)
                logits_output = model(input.copy())
                logits_output_ = model(input_.copy())
                # discretize the output graphs
                # binary_output = logits_output
                # binary_output_ = logits_output_
                
                # outputs_binary = (binary_output, binary_output_)
                outputs_logits = (logits_output, logits_output_)
                targets = (target, target_)

                loss = torch.tensor(0.0).to(device)
                if device is not None:
                    loss = loss.to(device)

                # loss is not averaged over the batch (single sample scale)
                # aet loss accepts gumbel-sigmoid output
                non_change_loss, change_loss = criterion(outputs_logits, targets, action_info)
                running_loss['non_change_val'].append(non_change_loss.item())
                loss += non_change_loss
                running_loss['change_val'].append(change_loss.item())
                loss += change_loss

                # statistics
                running_loss['val_loss'].append(loss.item())                

        avg_data_time = total_data_time / (idx+1)
        avg_model_time = total_model_time / (idx+1)
        running_loss = {k: np.mean(v) for k, v in running_loss.items()}
        # wandb.log({"aet_loss_avg_train": running_loss['aet_loss_train'], "superv_loss_avg_train": running_loss['superv_loss_train'],
        #              "train_loss_avg": running_loss['train_loss'], "aet_loss_avg_val": running_loss['aet_loss_val'],
        #                 "superv_loss_avg_val": running_loss['superv_loss_val'], "val_loss_avg": running_loss['val_loss'],
        #            "Epoch": epoch})

        total_loss_val = running_loss['val_loss']
        if total_loss_val < best_seen_running_validation_loss:
            best_seen_running_validation_loss = total_loss_val
            best_seen_model_weights = model.state_dict()
            logging.info(
                f"Found new best model with val loss "
                f"{best_seen_model_weights} at epoch {epoch}")
            
        total_loss_train = running_loss['train_loss']
        if total_loss_train < best_seen_model_train_loss:
            best_seen_model_train_loss = total_loss_train

    time_elapsed = time.perf_counter() - since
    num_min = time_elapsed // 60
    num_sec = time_elapsed % 60

    # if not do_validation:
    #     train_loss = running_loss['train']
    #     logging.info(f"Training complete in {num_min:.0f}m {num_sec:.0f}s "
    #                  f"with train loss {train_loss}")
    #     return model.state_dict()
    logging.info(
        f"Training complete in {num_min:.0f}m {num_sec:.0f}s "
        f"with train loss {best_seen_model_train_loss:.5f} and validation "
        f"loss {best_seen_running_validation_loss:.5f}")

    assert best_seen_model_weights
    return best_seen_model_weights

def calculate_entropy(prob_vector):
    # Ensure the probabilities are in the correct range
    prob_vector = torch.clamp(prob_vector, min=1e-10, max=1.0)
    entropy = -torch.sum(prob_vector * torch.log(prob_vector), dim=-1)
    return entropy

def convert_tgt_vector(tgt_vector):
    if tgt_vector.shape[-1] == 1:
        # no need to convert 0/1 matrix
        return tgt_vector
    else:
        assert tgt_vector.shape[-1] == 2, "tgt_matrix should have 2 channels"
        tgt_vector_new = tgt_vector[:, 0] * 1 + tgt_vector[:, 1] * 2
        assert tgt_vector_new.max() <= 2, "tgt_matrix_new should have max value 2, \
            got \n (Add) {} \n (Del) {}".format(tgt_vector[:,0], tgt_vector[:,1])
        return tgt_vector_new.unsqueeze(-1)

def compute_guidance_matrix(prob_matrix, tgt_matrix, entropy_w=0.5, loss_w=0.5, \
                            min_prob=0.0, max_prob=1.0):
    # Replace NaNs with zeros
    nan_mask = torch.isnan(prob_matrix)
    prob_matrix = torch.where(nan_mask, torch.tensor(0.0), prob_matrix)
    
    # Calculate entropy for each WxH position
    entropy_matrix = calculate_entropy(prob_matrix)
    
    # Set entropy to zero for positions where the original values were NaN
    nan_mask = torch.any(nan_mask, dim=-1)
    entropy_matrix = torch.where(nan_mask, torch.tensor(0.0), entropy_matrix)

    # Set other places (>0.0) to between neupi_entropy_entry_min and neupi_entropy_entry_max
    entropy_matrix[~nan_mask] = torch.clamp(entropy_matrix[~nan_mask], min=min_prob, max=max_prob)

    # Convert probability matrix to logits
    logits_matrix = torch.log(prob_matrix + 1e-10)
    
    # Flatten the matrices for loss computation
    tgt_matrix = convert_tgt_vector(tgt_matrix)
    logits_matrix_flat = logits_matrix.view(-1, logits_matrix.shape[-1])
    tgt_matrix_flat = tgt_matrix.view(-1)

    # Calculate cross-entropy loss for each entry
    loss_matrix = F.cross_entropy(logits_matrix_flat, tgt_matrix_flat, reduction='none')

    # Reshape the loss matrix back to WxH
    loss_matrix = loss_matrix.view(prob_matrix.shape[0], prob_matrix.shape[1])
    
    # Set loss to zero for positions where the original values were NaN
    loss_matrix = torch.where(nan_mask, torch.tensor(0.0, dtype=loss_matrix.dtype), loss_matrix)
    loss_matrix[~nan_mask] = torch.clamp(loss_matrix[~nan_mask], min=min_prob, max=max_prob)

    guidance_matrix = entropy_w*entropy_matrix + loss_w*loss_matrix
    
    return guidance_matrix

def extract_ae_vector(data: List[Tuple[State, Set[GroundAtom], State, Set[GroundAtom], _Option, str]], \
                      curr_candidate: Predicate,
                      ae_row_name_dict: Dict) -> torch.Tensor:
    ae_vector_dict = {}
    for k, v in ae_row_name_dict.items():
        ae_vector_dict[k] = [0, 0, 0] # [no_change, delete, add]

    for s, _, s_, _, opt, opt_info in data:
        action_id = opt_info.split('_')[0]
        action_id = int(action_id)
        act_objs = opt.objects
        pred_types = curr_candidate.types
        ground_objs = [None] * len(pred_types)
        for obj in act_objs:
            if obj.type in pred_types:
                pred_param_idx = pred_types.index(obj.type)
                ground_objs[pred_param_idx] = obj
        t_value = int(curr_candidate.holds(s, ground_objs))
        t_value_ = int(curr_candidate.holds(s_, ground_objs))
        # effect
        # no change
        if t_value == t_value_:
            ae_vector_dict[action_id][0] += 1
        # add
        elif t_value == 0 and t_value_ == 1:
            ae_vector_dict[action_id][1] += 1
        # delete
        elif t_value == 1 and t_value_ == 0:
            ae_vector_dict[action_id][2] += 1
        else:
            raise ValueError("Should be one of the three cases")
                
    ae_vector = torch.zeros([len(list(ae_row_name_dict.keys())), 3])
    for i in list(ae_row_name_dict.keys()):
        ae_vector[i, :] = torch.tensor(ae_vector_dict[i]) / \
            torch.tensor(ae_vector_dict[i]).sum()

    return ae_vector

def batched_node2types_gpt(nodes: torch.Tensor, node_feat_dict: Dict) -> List[str]:
    # Create a mapping from feature index to type
    feat_idx2type = {feat_idx: '_'.join(feat_name.split('_')[1:]) for feat_name, feat_idx in node_feat_dict.items() if "type" in feat_name}
    
    # Get the indices of the type features in the nodes matrix
    type_feature_indices = torch.tensor(list(feat_idx2type.keys()), device=nodes.device)
    
    # Extract the type channels using the type feature indices
    type_channels = nodes[:, type_feature_indices]
    
    # Find the first non-zero element along each row (which corresponds to the type)
    type_indices = torch.argmax(type_channels, dim=1)
    
    # Ensure there's at least one '1' in each row
    assert torch.all(type_channels.gather(1, type_indices.unsqueeze(1)) == 1), "Should have exactly one type per node"
    
    # Convert the indices back to their respective type names
    node_types = [feat_idx2type[type_feature_indices[idx].item()] for idx in type_indices]
    
    return node_types

def get_quantified_binary_edges(binary_output: Dict, binary_output_: Dict, 
                                action_ent: Dict,
                                node_type_graph: Dict, pred_type0: str, 
                                pred_type1: str, ent_idx: List[int], quantifier: str, quantified_type_idx: List[int],
                                w_negation: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    # get the types for the nodes in one sub graph
    # these node idx are for the predicate(?arg0, ?arg1)
    node_id0, node_id1 = None, None
    quantified_type = None
    if (pred_type1 not in node_type_graph) or (pred_type0 not in node_type_graph):
        # no nodes for this predicate
        return torch.tensor(0.0), torch.tensor(0.0)
    if not len(quantified_type_idx):
        assert len(ent_idx) == 2, "Should have 2 entities"
        ent0, ent1 = ent_idx[0], ent_idx[1]
        # no quantifier, directly use action_ent
        # find the operated object that is the argument in the pred
        if len(action_ent[pred_type0]) <= ent0:
            # action objs do not have enough objects for this predicate
            # random pick one
            assert len(node_type_graph[pred_type1]) > ent1
            node_id0 = [np.random.choice(node_type_graph[pred_type0])]
        else:
            node_id0 = [action_ent[pred_type0][ent0]]
        if len(action_ent[pred_type1]) <= ent1:
            # action objs do not have enough objects for this predicate
            # random pick one
            assert len(node_type_graph[pred_type0]) > ent0
            node_id1 = [np.random.choice(node_type_graph[pred_type1])]
        else:
            node_id1 = [action_ent[pred_type1][ent1]]
    elif len(quantified_type_idx) == 1:
        # quantify one type
        assert len(ent_idx) == 1, "Should have 1 entity"
        if quantified_type_idx[0] == 0:
            # quantified type is the first argument in predicate, 
            # check the second argument, 
            # then its ent idx in operator
            if len(action_ent[pred_type1]) == 1:
                # always take the only entity in action
                node_id1 = [action_ent[pred_type1][0]]
            elif len(action_ent[pred_type1]) >= ent_idx[0]+1:
                node_id1 = [action_ent[pred_type1][ent_idx[0]]]
            else:
                node_id1 = None
            # quantified type is the first type
            quantified_type = pred_type0
        else:
            # second argment, vice versa
            if len(action_ent[pred_type0]) == 1:
                # always take the only entity in action
                node_id0 = [action_ent[pred_type0][0]]
            elif len(action_ent[pred_type0]) >= ent_idx[0]+1:
                node_id0 = [action_ent[pred_type0][ent_idx[0]]]
            else:
                node_id0 = None
            quantified_type = pred_type1
        if (node_id0 is None) and (node_id1 is not None):
            # assert node_id1 is not None, "Should have node_id1"
            # node 0 is quantified, find all the corresponding types for node 0
            node_id0 = node_type_graph[quantified_type]
        elif (node_id1 is None) and (node_id0 is not None):
            # assert node_id0 is not None, "Should have node_id0"
            node_id1 = node_type_graph[quantified_type]
    else:
        # quantify two types
        assert len(quantified_type_idx) == 2, "Should have 2 quantified types"
        # all the predicate types are quntified, find all nodes
        node_id0 = node_type_graph[pred_type0]
        node_id1 = node_type_graph[pred_type1]
    if node_id0 is None or node_id1 is None:
        # no node pairs for the predicate
        return torch.tensor(0.0), torch.tensor(0.0)
    # both node 1 and node 2 are lists, so we need to use itertools.product
    sample_edges, sample_edges_ = [], []
    for n0, n1 in itertools.product(node_id0, node_id1):
        if n0 == n1:
            # no self loop
            continue
        edge_index = ((binary_output["senders"] == n0) & \
                        (binary_output["receivers"] == n1)).nonzero(as_tuple=True)[0]
        sample_edges.append(binary_output["edges"][edge_index[0]][0])
        
        edge_index_ = ((binary_output_["senders"] == n0) & \
                        (binary_output_["receivers"] == n1)).nonzero(as_tuple=True)[0]
        assert edge_index.shape[0] == 1, "Should have only one edge"
        assert torch.equal(edge_index, edge_index_), "Should be the same edge"
        sample_edges_.append(binary_output_["edges"][edge_index_[0]][0])

    if len(sample_edges) == 0:
        return torch.tensor(0.0), torch.tensor(0.0)
    
    # Not P(x, y) = 1 - P(x, y)
    if w_negation:
        sample_edges = 1 - torch.stack(sample_edges)
        sample_edges_ = 1 - torch.stack(sample_edges_)
    else:
        sample_edges = torch.stack(sample_edges)
        sample_edges_ = torch.stack(sample_edges_)
    # ForAll x P(x, y) = all comb P(x, y)
    if quantifier in ['', 'ForAll']:
        sample_edges = torch.all(sample_edges, dim=0).float()
        sample_edges_ = torch.all(sample_edges_, dim=0).float()
    else:
        # Exists x P(x, y) = any comb P(x, y)
        assert quantifier == 'Exists', "Should be Exists"
        sample_edges = torch.any(sample_edges, dim=0).float()
        sample_edges_ = torch.any(sample_edges_, dim=0).float()

    return sample_edges, sample_edges_

def inferece_dataloader(val_dataloader: torch.utils.data.DataLoader,
        pred_model: torch.nn.Module,
        gumbel_temp: float,
        boundary: float,
        device: Optional[torch.device] = None) -> List[Dict]:
    
    pred_model.train(False)
    pred_model.to(device)
    pred_model.eval()

    data_results = []
    with torch.no_grad():  # Disable gradient computation
        for data in val_dataloader:
            batched_data = copy.deepcopy(data)
            input = data['input']
            input_ = data['input_']
            # debug
            # target = data['target']
            # target_ = data['target_']

            # Move tensors to the correct device
            for key, val in input.items():
                input[key] = val.to(device) if val is not None else val
            for key, val in input_.items():
                input_[key] = val.to(device) if val is not None else val
            # for key, val in target.items():
            #     target[key] = val.to(device) if val is not None else val
            # for key, val in target_.items():
            #     target_[key] = val.to(device) if val is not None else val
            # Partition the input graph into discrete graphs
            logits_output = pred_model(input.copy())[-1]
            logits_output_ = pred_model(input_.copy())[-1]

            # outputs_logits = (logits_output, logits_output_)
            # targets = (target, target_)
            # for debug, checking the distribution of the output
            # non_change_loss, change_loss = neupi_supervise_criterion(
            #     outputs_logits, targets, None, CFG.neupi_super_label, device=device)

            # Discretize the output graphs
            binary_output = discrete_graph(logits_output, temperature=gumbel_temp, boundary=boundary)
            # prob_output = torch.sigmoid(logits_output['edges'] / gumbel_temp)
            binary_output_ = discrete_graph(logits_output_, temperature=gumbel_temp, boundary=boundary)
            # prob_output_ = torch.sigmoid(logits_output['edges'] / gumbel_temp)

            # Store relevant data for future use
            batched_data['binary_output'] = binary_output
            batched_data['binary_output_'] = binary_output_
            del logits_output, logits_output_
            torch.cuda.empty_cache()
            data_results.append(batched_data)
    return data_results

def distill_learned_ae_vector(val_dataloader: torch.utils.data.DataLoader,
                              gumbel_temp: float,
                              boundary: float,
                              pred_model: torch.nn.Module,
                              pred: DummyPredicate,
                              ent_idx: List[int],
                              ae_row_name_dict: Dict,
                              node_feat_dict: Dict,
                              device: Optional[torch.device] = None,
                              w_negation: Optional[bool] = False,
                              quantifier: Optional[str] = '',
                              quantified_type_idx: Optional[List] = []) -> torch.Tensor:
    ae_vector_dict = {}
    for k, v in ae_row_name_dict.items():
        ae_vector_dict[k] = [0, 0, 0] # [no_change, delete, add]

    pred_model.train(False)
    pred_model.to(device)

    for data in val_dataloader:
        input = data['input']
        input_ = data['input_']
        action_info = data['action_info']
        for key, val in input.items():
            input[key] = val.to(device) if val is not None else val
        for key, val in input_.items():
            input_[key] = val.to(device) if val is not None else val

        # the model partition the continous input graph into discrete graphs (groundings of predicates)
        node_types = batched_node2types_gpt(input['nodes'], node_feat_dict)
        if 'binary_output' in data:
            binary_output = data['binary_output']
            binary_output_ = data['binary_output_']
        else:
            logits_output = pred_model(input.copy())[-1]
            logits_output_ = pred_model(input_.copy())[-1]
            # discretize the output graphs
            binary_output = discrete_graph(logits_output, \
                                            temperature=gumbel_temp, \
                                            boundary=boundary)
            binary_output_ = discrete_graph(logits_output_, \
                                            temperature=gumbel_temp, \
                                            boundary=boundary)

        nodes = binary_output["nodes"]
        nodes_ = binary_output_["nodes"]
        s_node_id = 0

        # unary predicates
        if pred.arity == 1:
            pred_type = pred.types[0].name
            for idx, action_id in enumerate(action_info['action_id']):
                action_id = int(action_id)
                option_info = ae_row_name_dict[action_id]
                ent = ent_idx[0]
                act_node_ids = action_info['action_node_ids'][idx]
                action_types = option_info['types']
                action_ent = {
                    pred_type: []
                }
                node_id = None
                for acti_i, act_type in enumerate(action_types):
                    if act_type in action_ent:
                        action_ent[act_type].append(act_node_ids[acti_i].item())
                if len(action_ent[pred_type]) == 1:
                    node_id = action_ent[pred_type][0]
                # case two, action has multiple objects with the type
                elif len(action_ent[pred_type]) > 1:
                    # find the specified ent_idx object
                    node_id = action_ent[pred_type][ent]
                else:
                    # case three, action has no object with the type, all zero
                    logging.info("WARNING: No object with the type {} in action types {}".format(pred_type, action_types))
                    sample_nodes = torch.tensor(0)
                    sample_nodes_ = torch.tensor(0)
                if node_id is not None:
                    sample_nodes = nodes[node_id]
                    sample_nodes_ = nodes_[node_id]
                # effect
                # no change
                if torch.equal(sample_nodes, sample_nodes_):
                    ae_vector_dict[action_id][0] += 1
                # add
                elif torch.equal(sample_nodes, sample_nodes_ - 1):
                    ae_vector_dict[action_id][1] += 1
                # delete
                elif torch.equal(sample_nodes, sample_nodes_ + 1):
                    ae_vector_dict[action_id][2] += 1
                else:
                    raise ValueError("Should be one of the three cases")
        elif pred.arity == 2:
            # binary predicates
            pred_type0, pred_type1 = pred.types
            pred_type0 = pred_type0.name
            pred_type1 = pred_type1.name
            for idx, action_id in enumerate(action_info['action_id']):
                # get the types for the nodes in one sub graph
                e_node_id = s_node_id + input["n_node"][idx].item()
                node_type_graph = {}
                for nd_graph in range(s_node_id, e_node_id):
                    type_name = node_types[nd_graph]
                    if type_name not in node_type_graph:
                        node_type_graph[type_name] = [nd_graph]
                    else:
                        node_type_graph[type_name].append(nd_graph)

                s_node_id = e_node_id
                action_id = int(action_id)
                option_info = ae_row_name_dict[action_id]
                act_node_ids = action_info['action_node_ids'][idx]
                action_types = option_info['types']
                action_ent = {
                    pred_type0: [],
                    pred_type1: []
                }
                for acti_i, act_type in enumerate(action_types):
                    if act_type in action_ent:
                        action_ent[act_type].append(act_node_ids[acti_i].item())
                sample_edges, sample_edges_ = get_quantified_binary_edges(
                    binary_output, binary_output_, action_ent, node_type_graph, pred_type0, 
                    pred_type1, ent_idx, quantifier, quantified_type_idx, w_negation)
                # effect
                # no change
                if torch.equal(sample_edges, sample_edges_):
                    ae_vector_dict[action_id][0] += 1
                # add
                elif torch.equal(sample_edges, sample_edges_ - 1):
                    ae_vector_dict[action_id][1] += 1
                # delete
                elif torch.equal(sample_edges, sample_edges_ + 1):
                    ae_vector_dict[action_id][2] += 1
                else:
                    raise ValueError("Should be one of the three cases")
                
    ae_vector = torch.zeros([len(list(ae_row_name_dict.keys())), 3])
    for i in list(ae_row_name_dict.keys()):
        ae_vector[i, :] = torch.tensor(ae_vector_dict[i]) / \
            torch.tensor(ae_vector_dict[i]).sum()

    return ae_vector

def check_learned_ap_vector(val_dataloader: torch.utils.data.DataLoader,
                            gumbel_temp: float,
                            boundary: float,
                            pred_model: torch.nn.Module,
                            pred: DummyPredicate,
                            ent_idx: List[int],
                            ae_row_name_dict: Dict,
                            final_ae_vec: torch.Tensor,
                            node_feat_dict: Optional[Dict] = {},
                            device: Optional[torch.device] = None,
                            w_negation: Optional[bool] = False,
                            quantifier: Optional[str] = '',
                            quantified_type_idx: Optional[List] = [],
                            thresh: Optional[float] = 1.0) -> torch.Tensor:
    logging.info("Checking the learned AP vector...")
    logging.info("Using the provided AE vector (Add): {}".format(final_ae_vec[:, 0]))
    if final_ae_vec.shape[1] == 2:
        logging.info("Using the provided AE vector (Del): {}".format(final_ae_vec[:, 1]))
    if (final_ae_vec[:, 0].sum() == 0) or (final_ae_vec[:, 1].sum() == 0):
        # delete-only or add-only effect predicates
        ap_vec = torch.zeros([len(ae_row_name_dict.keys()), 2]) # 0: exists, 1: not exists
        
        pred_model.train(False)
        pred_model.to(device)

        for data in val_dataloader:
            input = data['input']
            input_ = data['input_']
            action_info = data['action_info']
            for key, val in input.items():
                input[key] = val.to(device) if val is not None else val
            for key, val in input_.items():
                input_[key] = val.to(device) if val is not None else val
            
            # the model partition the continous input graph into discrete graphs (groundings of predicates)
            node_types = batched_node2types_gpt(input['nodes'], node_feat_dict)
            if 'binary_output' in data:
                binary_output = data['binary_output']
                binary_output_ = data['binary_output_']
            else:
                logits_output = pred_model(input.copy())[-1]
                logits_output_ = pred_model(input_.copy())[-1]
                # discretize the output graphs
                binary_output = discrete_graph(logits_output, \
                                                temperature=gumbel_temp, \
                                                boundary=boundary)
                binary_output_ = discrete_graph(logits_output_, \
                                                temperature=gumbel_temp, \
                                                boundary=boundary)
            
            nodes = binary_output["nodes"]
            nodes_ = binary_output_["nodes"]
            s_node_id = 0

            # unary predicates
            if pred.arity == 1:
                pred_type = pred.types[0].name
                for idx, action_id in enumerate(action_info['action_id']):
                    action_id = int(action_id)
                    option_info = ae_row_name_dict[action_id]
                    ent = ent_idx[0]
                    act_node_ids = action_info['action_node_ids'][idx]
                    action_types = option_info['types']
                    action_ent = {
                        pred_type: []
                    }
                    node_id = None
                    for acti_i, act_type in enumerate(action_types):
                        if act_type in action_ent:
                            action_ent[act_type].append(act_node_ids[acti_i])
                    if len(action_ent[pred_type]) == 1:
                        node_id = action_ent[pred_type][0]
                    # case two, action has multiple objects with the type
                    elif len(action_ent[pred_type]) > 1:
                        # find the specified ent_idx object
                        node_id = action_ent[pred_type][ent]
                    else:
                        # case three, action has no object with the type, all zero
                        logging.info("WARNING: No object with the type {} in action types {}".format(pred_type, action_types))
                        sample_nodes = torch.tensor(0)
                        sample_nodes_ = torch.tensor(0)
                    if node_id is not None:
                        sample_nodes = nodes[node_id]
                        sample_nodes_ = nodes_[node_id]
                    if equiv_effect(sample_nodes, sample_nodes_, final_ae_vec[action_id]):
                        # precondition
                        if sample_nodes:
                            ap_vec[action_id, 0] += 1
                        else:
                            ap_vec[action_id, 1] += 1
                    else:
                        logging.info("WARNING: Found sample that is not equivalent to AE Vector")
            elif pred.arity == 2:
                # binary predicates
                pred_type0, pred_type1 = pred.types
                pred_type0 = pred_type0.name
                pred_type1 = pred_type1.name
                for idx, action_id in enumerate(action_info['action_id']):
                    # get the types for the nodes in one sub graph
                    e_node_id = s_node_id + input["n_node"][idx].item()
                    node_type_graph = {}
                    for nd_graph in range(s_node_id, e_node_id):
                        type_name = node_types[nd_graph]
                        if type_name not in node_type_graph:
                            node_type_graph[type_name] = [nd_graph]
                        else:
                            node_type_graph[type_name].append(nd_graph)

                    s_node_id = e_node_id
                    action_id = int(action_id)
                    option_info = ae_row_name_dict[action_id]
                    act_node_ids = action_info['action_node_ids'][idx]
                    action_types = option_info['types']
                    action_ent = {
                        pred_type0: [],
                        pred_type1: []
                    }
                    for acti_i, act_type in enumerate(action_types):
                        if act_type in action_ent:
                            action_ent[act_type].append(act_node_ids[acti_i].item())
                    sample_edges, sample_edges_ = get_quantified_binary_edges(
                        binary_output, binary_output_, action_ent, node_type_graph, pred_type0, 
                        pred_type1, ent_idx, quantifier, quantified_type_idx, w_negation)
                    if equiv_effect(sample_edges, sample_edges_, final_ae_vec[action_id]):
                        # precondition
                        if sample_edges:
                            ap_vec[action_id, 0] += 1
                        else:
                            ap_vec[action_id, 1] += 1
                    else:
                        logging.info("WARNING: Found sample that is not equivalent to AE Vector")
    
        return precondition_checker(ap_vec, final_ae_vec, thresh)
    else:
        # informative! it should appear in the preconditions of another operator
        logging.info("Both Del and Add Predicate! Informative Passed!!!")
        return True

def equiv_effect(sample, sample_, ae_vec) -> bool:
    if torch.equal(sample, sample_):
        # no change
        return ae_vec.all() == 0
    elif torch.equal(sample, sample_ - 1):
        # add effect
        return ae_vec[0] > 0.9
    elif torch.equal(sample, sample_ + 1):
        # delete effect
        return ae_vec[1] > 0.9
    else:
        raise ValueError("Should be one of the three cases")
    
def precondition_checker(ap_vec, final_ae_vec, thresh=0.99) -> torch.Tensor:
    # it must appear in the preconditions of another operator
    non_effect_idx = torch.where(final_ae_vec.sum(dim=1) == 0)[0]
    extracted_preconditions = ap_vec[non_effect_idx, 0] / ap_vec[non_effect_idx].sum(dim=1)
    logging.info("Del/Add Only Predicate. Checking Rows (Ops): {}".format(non_effect_idx))
    logging.info("Using Threshold: {}".format(thresh))
    if not (extracted_preconditions >= thresh).any():
        logging.info("Failed!!! Resulting Pre-condition: {}" \
                        .format(ap_vec[:, 0] / ap_vec.sum(dim=1)))
    else:
        logging.info("Passed!!!")
    return (extracted_preconditions >= thresh).any()
        
def compute_guidance_vector(prob_vector, tgt_vector=None, \
                            entropy_w=0.5, loss_w=0.5, \
                            min_prob=0.0, max_prob=1.0):
    # Replace NaNs with zeros
    nan_mask = torch.isnan(prob_vector)
    prob_vector = torch.where(nan_mask, torch.tensor(0.0), prob_vector)
    
    # Calculate entropy for each WxH position
    entropy_vector = calculate_entropy(prob_vector)
    
    # Set entropy to zero for positions where the original values were NaN
    nan_mask = torch.any(nan_mask, dim=-1)
    entropy_vector = torch.where(nan_mask, torch.tensor(0.0), entropy_vector)

    # Set other places (>0.0) to between neupi_entropy_entry_min and neupi_entropy_entry_max
    entropy_vector[~nan_mask] = torch.clamp(entropy_vector[~nan_mask], min=min_prob, max=max_prob)

    # Convert probability matrix to logits
    if tgt_vector is not None:
        logits_vector = torch.log(prob_vector + 1e-10)
        
        # Flatten the matrices for loss computation
        tgt_vector = convert_tgt_vector(tgt_vector)
        logits_vec_flat = logits_vector.view(-1, logits_vector.shape[-1])
        tgt_vec_flat = tgt_vector.view(-1)

        # Calculate cross-entropy loss for each entry
        loss_vec = F.cross_entropy(logits_vec_flat, tgt_vec_flat, reduction='none')

        # Reshape the loss matrix back to WxH
        loss_vec = loss_vec.view(prob_vector.shape[0])
        
        # Set loss to zero for positions where the original values were NaN
        loss_vec = torch.where(nan_mask, torch.tensor(0.0, dtype=loss_vec.dtype), loss_vec)
        loss_vec[~nan_mask] = torch.clamp(loss_vec[~nan_mask], min=min_prob, max=max_prob)
    else:
        entropy_w = 1.0
        loss_w = 0.0
        loss_vec = torch.zeros_like(entropy_vector)

    guidance_vec = entropy_w*entropy_vector + loss_w*loss_vec
    return guidance_vec
