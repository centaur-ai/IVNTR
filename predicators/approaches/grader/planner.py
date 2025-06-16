from copy import deepcopy
import numpy as np
import os

import torch
import logging
import torch.nn as nn
import networkx as nx

from predicators.approaches.grader.gnn_modules import RGCN, MLP, GRU_SCM
from predicators.approaches.grader.grader_utils import CUDA, kaiming_init


class WorldModel(object):
    def __init__(self, args):
        self.state_dim = args['env_params']['state_dim']
        self.action_dim = args['env_params']['action_dim']
        self.env_name = args['env_params']['env_name']
        self.grader_model = args['grader_model']
        self.use_discover = args['use_discover']

        assert self.grader_model in ['causal', 'full', 'mlp', 'offline', 'gnn']

        self.n_epochs = args['n_epochs']
        self.lr = args['lr']
        self.batch_size = args['batch_size']

        self.validation_flag = args['validation_flag']
        self.validate_freq = args['validation_freq']
        self.validation_ratio = args['validation_ratio']
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        # process things that are different in environments
        if self.env_name == 'satellites_simple':
            self.build_node_and_edge = self.build_node_and_edge_satellites
            self.organize_nodes = self.organize_nodes_satellites
            # self.adjacency_matrix = args['env_params']['adjacency_matrix']
            self.num_ent = args['env_params']['num_sat'] + args['env_params']['num_obj']
            # self.adjacency_matrix += np.eye(self.adjacency_matrix.shape[0]) # add diagonal elements
            #
            self.num_obj_pairs = self.num_ent * (self.num_ent - 1)
            self.oc_dim = sum([args['env_params']['obj_centric_feat_dim']] * self.num_ent)
            self.rel_dim = sum([args['env_params']['rel_feat_dim']] * self.num_obj_pairs)
            self.state_dim_list = [args['env_params']['obj_centric_feat_dim']] * self.num_ent + \
                        [args['env_params']['rel_feat_dim']] * self.num_obj_pairs
            assert sum(self.state_dim_list) == self.state_dim
            self.action_dim_list = [args['env_params']['action_dim']] # action does not have causal variables
        elif self.env_name == 'blocks_simple':
            self.build_node_and_edge = self.build_node_and_edge_satellites
            self.organize_nodes = self.organize_nodes_satellites
            self.num_ent = args['env_params']['num_blocks'] + 1
            self.num_obj_pairs = self.num_ent * (self.num_ent - 1)
            self.state_dim_list = [args['env_params']['obj_centric_feat_dim']] * self.num_ent + \
                        [args['env_params']['rel_feat_dim']] * self.num_obj_pairs
            self.action_dim_list = [args['env_params']['action_dim']]
            self.oc_dim = sum([args['env_params']['obj_centric_feat_dim']] * self.num_ent)
            self.rel_dim = sum([args['env_params']['rel_feat_dim']] * self.num_obj_pairs)
        else:
            raise ValueError('Unknown environment name')

        self.use_full = False
        self.use_mlp = False
        if self.grader_model == 'mlp':
            self.model_name = 'mlp'
            self.use_mlp = True
        elif self.grader_model == 'causal':
            self.model_name = 'gru'
        elif self.grader_model == 'full':
            self.model_name = 'gru'
            self.use_full = True
        elif self.grader_model == 'gnn':
            self.model_name = 'gnn'
            self.use_full = True

        random = False
        if self.model_name == 'mlp':
            input_dim = self.state_dim + self.action_dim
            output_dim = self.state_dim
            self.model = CUDA(MLP(input_dim, output_dim, args["hidden_dim"], args["hidden_size"], dropout_p=0.0))
            hidden_dim = args["hidden_size"]
        elif self.model_name == 'gru' or self.model_name == 'gnn':
            edge_dim = 1
            hidden_num = 1
            if self.env_name == 'satellites_simple':
                args["hidden_dim"] = 64
                
            hidden_dim = args["hidden_dim"]
            self.node_num = len(self.action_dim_list) + len(self.state_dim_list)
            self.node_dim = int(np.max(self.state_dim_list+self.action_dim_list))
            if self.model_name == 'gnn':
                self.model = CUDA(RGCN(self.node_dim, self.node_num, 'mean', args["hidden_dim"], self.node_dim, edge_dim, hidden_num))
            else:
                self.model = CUDA(GRU_SCM(self.action_dim_list, self.state_dim_list, self.node_num, 'mean', args["hidden_dim"], edge_dim, hidden_num, dropout=0.0, random=random))

        print('----------------------------')
        print('Env:', self.env_name)
        print('GRADER model:', self.grader_model)
        print('Model_name:', self.model_name)
        print('Full:', self.use_full)
        print('SCM noise:', random)
        print('Hidden dim:', hidden_dim)
        print('----------------------------')

        self.model.apply(kaiming_init)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.buffer_length = 0
        self.criterion = self.mse_loss
        # self.rel_criterion = self.ce_loss

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.data = None
        self.label = None
        self.eps = 1e-30

        if self.grader_model == 'causal':
            # the initial graph is a lower triangular graph
            self.causal_graph = np.zeros((self.node_num, self.node_num))
            for i in range(self.causal_graph.shape[0]):
                for j in range(self.causal_graph.shape[1]):
                    if i >= j:
                        self.causal_graph[i, j] = 1
        self.best_test_loss = np.inf

    def build_node_and_edge_satellites(self, data):
        # create the node matrix. the last node is the output node therefore should always be 0.
        batch_size = data.shape[0]
        x = torch.zeros((batch_size, self.node_num, self.node_dim), device=torch.device(self.device)) # [B, 125]

        # build the nodes of action
        action = data[:, sum(self.state_dim_list):]
        start_ = 0
        for a_i in range(len(self.action_dim_list)):
            end_ = self.action_dim_list[a_i] + start_
            x[:, a_i, 0:end_-start_] = action[:, start_:end_] # pad 0 for remaining places
            start_ = end_

        # build the nodes of state
        state = data[:, 0:sum(self.state_dim_list)]

        # [B, N*C*W*H] -> [B, N, C*W*H]
        oc_state = state[:, 0:self.oc_dim]
        oc_state = oc_state.reshape(batch_size, self.num_ent, -1)
        rel_state = state[:, self.oc_dim:]
        rel_state = rel_state.reshape(batch_size, self.num_obj_pairs, -1)
        start_ = 0
        for s_i in range(self.num_ent):
            end_ = self.state_dim_list[s_i] + start_
            x[:, s_i+len(self.action_dim_list), 0:end_-start_] = oc_state[:, s_i, :] # pad 0 for remaining places
            start_ = end_
        for s_i in range(self.num_obj_pairs):
            end_ = self.state_dim_list[self.num_ent+s_i] + start_
            x[:, s_i+len(self.action_dim_list)+self.num_ent, 0:end_-start_] = rel_state[:, s_i, :]
            start_ = end_

        if self.use_full:
            # full graph (states are fully connected)
            full = np.ones((self.node_num, self.node_num))
            action_row = np.zeros((1, self.node_num))
            action_row[0] = 1
            full[0, :] = action_row
            adj = full

        if self.use_discover:
            adj = self.causal_graph

        # if self.use_gt:
        #     # using GT causal graph
        #     gt_adj = np.zeros((self.node_num, self.node_num))
        #     gt_adj[1:, 1:] = self.adjacency_matrix
        #     gt_adj[:, 0] = 1.0
        #     adj = gt_adj

        adj = np.array(adj)[None, None, :, :].repeat(batch_size, axis=0)
        adj = CUDA(torch.from_numpy(adj.astype(np.float32)))
        return x, adj

    def organize_nodes_satellites(self, x):
        # x - [B, node_num, node_dim], the nodes of next_state are in the end
        delta_state_node = x[:, -len(self.state_dim_list):, :]
        delta_state_oc = []
        for s_i in range(self.num_ent):
            state_i = delta_state_node[:, s_i:s_i+1, 0:self.state_dim_list[s_i]] 
            delta_state_oc.append(state_i)
        delta_state_oc = torch.cat(delta_state_oc, dim=1) # [B, N, D]
        delta_state_oc = delta_state_oc.reshape(delta_state_oc.shape[0], -1)
        delta_state_rel = []
        for s_i in range(self.num_obj_pairs):
            state_i = delta_state_node[:, s_i+self.num_ent:self.num_ent+s_i+1, 0:self.state_dim_list[self.num_ent+s_i]]
            delta_state_rel.append(state_i)
        delta_state_rel = torch.cat(delta_state_rel, dim=1) # [B, N, D']
        delta_state_rel = delta_state_rel.reshape(delta_state_rel.shape[0], -1)
        delta_state = torch.cat([delta_state_oc, delta_state_rel], dim=1)
        return delta_state

    def data_process(self, data, max_buffer_size):
        x = data[0][None]
        label = data[1][None]
        self.buffer_length += 1
    
        # add new data point to data buffer
        if self.data is None:
            self.data = CUDA(torch.from_numpy(x.astype(np.float32)))
            self.label = CUDA(torch.from_numpy(label.astype(np.float32)))
        else:
            if self.data.shape[0] < max_buffer_size:
                self.data = torch.cat((self.data, CUDA(torch.from_numpy(x.astype(np.float32)))), dim=0)
                self.label = torch.cat((self.label, CUDA(torch.from_numpy(label.astype(np.float32)))), dim=0)
            else:
                # replace the old buffer
                #index = self.buffer_length % max_buffer_size # sequentially replace buffer
                index = np.random.randint(0, max_buffer_size) # randomly replace buffer
                self.data[index] = CUDA(torch.from_numpy(x.astype(np.float32)))
                self.label[index] = CUDA(torch.from_numpy(label.astype(np.float32)))

    def split_train_validation(self):
        num_data = len(self.data)

        # use validation
        if self.validation_flag:
            indices = list(range(num_data))
            split = int(np.floor(self.validation_ratio * num_data))
            np.random.shuffle(indices)
            if self.validation_ratio < 1:
                train_idx, test_idx = indices[split:], indices[:split]
            else:
                # debug mode, use all data for training
                train_idx, test_idx = indices, indices

            train_set = [[self.data[idx], self.label[idx]] for idx in train_idx]
            test_set = [[self.data[idx], self.label[idx]] for idx in test_idx]

            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
        else:
            train_set = [[self.data[idx], self.label[idx]] for idx in range(num_data)]
            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = None
        return train_loader, test_loader

    def fit(self):
        self.model.train()
        train_loader, test_loader = self.split_train_validation()

        self.best_test_loss = np.inf
        saved_stats = []
        for epoch in range(self.n_epochs):
            running_stats = {'loss': [], 'zero_loss': [], 'non_zero_loss': []}
            for datas, labels in train_loader:
                self.optimizer.zero_grad()
                zero_label = labels == 0
                non_zero_label = labels != 0

                if self.use_mlp:
                    delta = self.model(datas)
                    zero_loss = self.criterion(delta[zero_label], labels[zero_label])
                    non_zero_loss = self.criterion(delta[non_zero_label], labels[non_zero_label])
                    loss = zero_loss + non_zero_loss
                else:
                    x, adj = self.build_node_and_edge(datas)
                    x = self.model(x, adj)
                    delta = self.organize_nodes(x)
                    zero_loss = self.criterion(delta[zero_label], labels[zero_label])
                    non_zero_loss = self.criterion(delta[non_zero_label], labels[non_zero_label])
                    loss = zero_loss + non_zero_loss
                running_stats['zero_loss'].append(zero_loss.item())
                running_stats['non_zero_loss'].append(non_zero_loss.item())
                running_stats['loss'].append(loss.item())
                loss.backward()
                self.optimizer.step()
            running_stats['loss'] = np.mean(running_stats['loss'])
            running_stats['zero_loss'] = np.mean(running_stats['zero_loss'])
            running_stats['non_zero_loss'] = np.mean(running_stats['non_zero_loss'])
            # logging.info('Epoch: %d, Loss: %.4f, Zero Loss: %.4f, Non Zero Loss: %.4f' 
            #              % (epoch, running_stats['loss'], running_stats['zero_loss'], running_stats['non_zero_loss']))

            if self.validation_flag and (epoch+1) % self.validate_freq == 0:
                with torch.no_grad():
                    loss_test = self.validate_model(test_loader)
                saved_stats.append([running_stats['loss'], loss_test])
                if loss_test < self.best_test_loss:
                    self.best_test_loss = loss_test
                    logging.info('New best validation Loss: %.4f' % loss_test)
                    self.best_model = deepcopy(self.model.state_dict())
        # save the stats for visualization
        if self.validation_flag:
            saved_stats = np.array(saved_stats)
            np.save('saved_stats.npy', saved_stats)
        # load the best model if we use validation
        if self.validation_flag:
            self.model.load_state_dict(self.best_model)
        return self.best_test_loss

    def validate_model(self, testloader):
        self.model.eval()
        loss_list = []
        for datas, labels in testloader:
            zero_label = labels == 0
            non_zero_label = labels != 0
            if self.use_mlp:
                delta = self.model(datas)
                loss = self.criterion(delta, labels)
                zero_loss = self.criterion(delta[zero_label], labels[zero_label])
                non_zero_loss = self.criterion(delta[non_zero_label], labels[non_zero_label])
                loss = zero_loss + non_zero_loss
            else:
                x, adj = self.build_node_and_edge(datas)
                x = self.model(x, adj)
                delta = self.organize_nodes(x)
                zero_loss = self.criterion(delta[zero_label], labels[zero_label])
                non_zero_loss = self.criterion(delta[non_zero_label], labels[non_zero_label])
                loss = zero_loss + non_zero_loss

            loss_list.append(loss.item())
        self.model.train()
        return np.mean(loss_list)

    def predict(self, s, a):
        self.model.eval()
        # convert to torch format
        if isinstance(s, np.ndarray):
            s = CUDA(torch.from_numpy(s.astype(np.float32)))
        if isinstance(a, np.ndarray):
            a = CUDA(torch.from_numpy(a.astype(np.float32)))

        inputs = torch.cat((s, a), axis=1)

        with torch.no_grad():
            if self.use_mlp:
                delta = self.model(inputs)
            else:
                x, adj = self.build_node_and_edge(inputs)
                x = self.model(x, adj)
                delta = self.organize_nodes(x)

            delta = delta.cpu().detach().numpy()
        return delta

    def save_model(self, info):
        states = {'wm_state_dict': self.model.state_dict()}
        info.update(states)
        return info

    def load_model(self, info):
        state_dict = info['wm_state_dict']
        self.model.load_state_dict(state_dict)


class Planner(object):
    def __init__(self, args):
        self.pretrain_buffer_size = args['pretrain_buffer_size']
        self.max_buffer_size = args['max_buffer_size']
        self.epsilon = args['epsilon']
        self.model = WorldModel(args)

    def store_transition(self, data):
        # [state, action, next_state]
        # we should remove the goal infomation from x and label
        pure_state = data[0][:len(data[0])]
        action = data[1]
        pure_next_state = data[2][:len(data[0])]
        x = np.concatenate([pure_state, action])
        label = pure_next_state - pure_state 
        self.model.data_process([x, label], self.max_buffer_size)

    def train(self):
        # when data has been collected enough, train model
        if self.model.data.shape[0] < self.pretrain_buffer_size:
            self.best_test_loss = 0
        else:
            self.best_test_loss = self.model.fit()

    def set_causal_graph(self, causal_graph):
        self.model.causal_graph = causal_graph

    def save_model(self, info):
        return self.model.save_model(info)

    def load_model(self, info):
        self.model.load_model(info)
