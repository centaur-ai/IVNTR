"""An approach that trains a Transformer mapping states, atoms, and goals to
dummy NSRTs, with GT sampler."""

import time
import logging
import itertools
from collections import defaultdict
from typing import Optional, Dict, List, Set, Tuple

import numpy as np
import dill as pkl
import torch
import torch.nn
import torch.optim
from gym.spaces import Box
from torch.utils.data import DataLoader

from predicators import utils
from predicators.approaches.gnn_nsrt_policy_approach import GNNNSRTPolicyApproach
from predicators.gnn.gnn_utils import GraphDictDataset, compute_normalizers, \
    graph_batch_collate, normalize_graph, \
    train_model
from predicators.gnn.tf import setup_pointnet_tf_net, setup_tf_net
from predicators.settings import CFG
from predicators.gnn.gnn_utils import normalize_graph
from predicators.structs import Action, Array, Dataset, DummyOption, \
    GroundAtom, NSRT, ParameterizedOption, Predicate, State, Task, Type, \
    _Option

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

class TransformerNSRTPolicyApproach(GNNNSRTPolicyApproach):
    """Trains and uses a goal-conditioned GNN policy that produces options."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
    
    @classmethod
    def get_name(cls) -> str:
        return "transformer_nsrt_policy"

    def load(self, online_learning_cycle: Optional[int]) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.gnn", "rb") as f:
            info = pkl.load(f)
        # Initialize fields from loaded dictionary.
        input_example, target_example = info["exemplar"]
        dataset = GraphDictDataset([input_example], [target_example])
        self._nullary_predicates = info["nullary_predicates"]
        self._node_feature_to_index = info["node_feature_to_index"]
        self._edge_feature_to_index = info["edge_feature_to_index"]
        self._input_normalizers = info["input_normalizers"]
        self._target_normalizers = info["target_normalizers"]
        if CFG.gnn_use_pointnet:
            self._gnn = setup_pointnet_tf_net(dataset,
                                              tf_dim=CFG.gnn_layer_size,
                                              pointnet_feat_lst=CFG.gnn_pointnet_feat_lst,
                                              node_feature_to_index=self._node_feature_to_index.copy(),
                                              gnn_layer_size=CFG.gnn_layer_size,
                                              num_layers=CFG.gnn_num_layers)
        else:
            self._gnn = setup_tf_net(dataset,
                                    tf_dim=CFG.gnn_layer_size,
                                    gnn_layer_size=CFG.gnn_layer_size,
                                    num_layers=CFG.gnn_num_layers)
        self._gnn.load_state_dict(info["state_dict"])
        self._load_output_specific_fields_from_save_info(info)

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        data = self._generate_data_from_dataset(dataset)
        self._setup_fields(data)
        # Set up exemplar, which is just the first tuple in the data.
        example_input, example_object_to_node = self._graphify_single_input(
            data[0][0], data[0][1], data[0][2])
        example_target = self._graphify_single_target(data[0][3],
                                                      example_input,
                                                      example_object_to_node)
        self._data_exemplar = (example_input, example_target)
        example_dataset = GraphDictDataset([example_input], [example_target])
        if CFG.gnn_use_pointnet:
            self._gnn = setup_pointnet_tf_net(example_dataset,
                                              tf_dim=CFG.gnn_layer_size,
                                              pointnet_feat_lst=CFG.gnn_pointnet_feat_lst,
                                              node_feature_to_index=self._node_feature_to_index.copy(),
                                              gnn_layer_size=CFG.gnn_layer_size,
                                              num_layers=CFG.gnn_num_layers)
        else:
            self._gnn = setup_tf_net(example_dataset,
                                    tf_dim=CFG.gnn_layer_size,
                                    gnn_layer_size=CFG.gnn_layer_size,
                                    num_layers=CFG.gnn_num_layers)
        # Set up all the graphs, now using *all* the data.
        inputs = [(d[0], d[1], d[2]) for d in data]
        targets = [d[3] for d in data]
        graph_inputs = []
        graph_targets = []
        for (state, atoms, goal), target in zip(inputs, targets):
            graph_input, object_to_node = self._graphify_single_input(
                state=state, atoms=atoms, goal=goal)
            graph_inputs.append(graph_input)
            graph_targets.append(
                self._graphify_single_target(target, graph_input,
                                             object_to_node))
        if CFG.gnn_do_normalization:
            # Update normalization constants. Note that we do this for both
            # the input graph and the target graph.
            self._input_normalizers = compute_normalizers(graph_inputs, 
                                        normalize_nodes=self._node_is_rot, \
                                        normalize_edges=self._edge_is_rot, \
                                        normalize_globals=False)
            tgt_node_indicators = [False] * graph_targets[0]["nodes"].shape[1]
            tgt_edge_indicators = [False] * graph_targets[0]["edges"].shape[1]
            self._target_normalizers = compute_normalizers(graph_targets, 
                                        normalize_nodes=tgt_node_indicators, \
                                        normalize_edges=tgt_edge_indicators, \
                                        normalize_globals=False)
            graph_inputs = [
                normalize_graph(g, self._input_normalizers)
                for g in graph_inputs
            ]
            graph_targets = [
                normalize_graph(g, self._target_normalizers)
                for g in graph_targets
            ]
        # Run training.
        if CFG.gnn_use_validation_set:
            ## Split data, using 10% for validation.
            num_validation = max(1, int(len(inputs) * 0.1))
        else:
            num_validation = 0
        if not CFG.neupi_bug:
            train_inputs = graph_inputs[num_validation:]
            train_targets = graph_targets[num_validation:]
            val_inputs = graph_inputs[:num_validation]
            val_targets = graph_targets[:num_validation]
        else:
            logging.info("Running in debug mode; Val will be the same as train.")
            train_inputs = graph_inputs
            train_targets = graph_targets
            # random sample 10% of the data for validation
            val_ids = np.random.choice(len(graph_inputs), num_validation, replace=False)
            val_inputs = [graph_inputs[i] for i in val_ids]
            val_targets = [graph_targets[i] for i in val_ids]
        train_dataset = GraphDictDataset(train_inputs, train_targets)
        val_dataset = GraphDictDataset(val_inputs, val_targets)
        ## Set up Adam optimizer and dataloaders.
        optimizer = torch.optim.Adam(self._gnn.parameters(),
                                     lr=CFG.gnn_learning_rate)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CFG.gnn_batch_size,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      persistent_workers=True,
                                      collate_fn=graph_batch_collate)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=graph_batch_collate)
        dataloaders = {"train": train_dataloader, "val": val_dataloader}
        ## Launch training code.
        logging.info(f"Training GNN on {len(train_inputs)} examples using {CFG.device} device")
        best_model_dict = train_model(self._gnn,
                                      dataloaders,
                                      optimizer=optimizer,
                                      criterion=self._criterion,
                                      global_criterion=self._global_criterion,
                                      num_epochs=CFG.gnn_num_epochs,
                                      do_validation=CFG.gnn_use_validation_set,
                                      device=CFG.device)
        self._gnn.load_state_dict(best_model_dict)
        info = {
            "exemplar": self._data_exemplar,
            "state_dict": self._gnn.state_dict(),
            "nullary_predicates": self._nullary_predicates,
            "node_feature_to_index": self._node_feature_to_index,
            "edge_feature_to_index": self._edge_feature_to_index,
            "input_normalizers": self._input_normalizers,
            "target_normalizers": self._target_normalizers,
        }
        self._add_output_specific_fields_to_save_info(info)
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_None.gnn", "wb") as f:
            pkl.dump(info, f)