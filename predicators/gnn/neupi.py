"""Graph neural network core code.

A lot of this was copied and modified from code originally written by
Kelsey Allen.
"""

import abc
import logging
from typing import Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from predicators.gnn.gnn_utils import GraphTransC2DDataset, replace_graph
from predicators.gnn.resnet_utils import ResNet18Encoder
from predicators.gnn.pointnet_utils import PointNetEncoder
from pytorch3d.transforms import quaternion_to_matrix

def array_in_list(array, list_of_arrays):
    return any(np.array_equal(array, x) for x in list_of_arrays)

class ResNetMLPClassifier(nn.Module):
    def __init__(self, img_feat_ind,
                 encoder,
                 init_fn):
        super(PointNetMLPClassifier, self).__init__()
        self.arity = 2
        self.img_feat_ind = img_feat_ind
        self.vec_enc = encoder
        self.feat = ResNet18Encoder()
        self.mlp_encoder = encoder
        self.apply_init(init_fn)

    def forward(self, graph: Dict) -> List[Dict]:
        input_data = graph['nodes']
        B = input_data.shape[0]
        input_data_img = input_data[:, self.img_feat_ind]
        img = input_data_img.view(B, -1, 3)
        decoded = self.feat(img)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph

    def apply_init(self, init_fn):
        """
        Apply custom initialization to all submodules, including encoders and linear layers.
        Parameters:
            init_fn (callable): A function to initialize the weights of layers.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Apply custom initialization to weights
                init_fn(module.weight)
                if module.bias is not None:
                    # Initialize biases uniformly
                    nn.init.uniform_(module.bias, -1.0, 1.0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Initialize BatchNorm layers
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

class PointNetMLPClassifier(nn.Module):
    def __init__(self, pcd_feat_ind, vec_feat_ind,
                 encoder,
                 encoder_last_dim,
                 pointnet_ls_list,
                 init_fn):
        super(PointNetMLPClassifier, self).__init__()
        self.arity = 2
        self.pcd_feat_ind = pcd_feat_ind
        self.vec_feat_ind = vec_feat_ind
        self.vec_enc = encoder
        self.feat = PointNetEncoder(pointnet_ls_list, \
                                    global_feat=True, \
                                    feature_transform=True)
        self.fc1 = nn.Linear(pointnet_ls_list[2] + encoder_last_dim, \
                             pointnet_ls_list[1])
        self.fc2 = nn.Linear(pointnet_ls_list[1], pointnet_ls_list[0])
        self.fc3 = nn.Linear(pointnet_ls_list[0], 1)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(pointnet_ls_list[1])
        self.bn2 = nn.BatchNorm1d(pointnet_ls_list[0])
        self.mlp_encoder = encoder
        self.apply_init(init_fn)

    def forward(self, graph: Dict) -> List[Dict]:
        input_data = graph['edges']
        B = input_data.shape[0]
        input_data_pcd = input_data[:, self.pcd_feat_ind]
        xyz = input_data_pcd.view(B, -1, 3)
        xyz = xyz.permute(0, 2, 1)  # xyz: B x 3 x N
        x = self.feat(xyz)
        input_data_vec = input_data[:, self.vec_feat_ind]
        vec_encoded = self.vec_enc(input_data_vec)
        x = torch.cat([x, vec_encoded], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        decoded = self.fc3(x)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph

    def apply_init(self, init_fn):
        """
        Apply custom initialization to all submodules, including encoders and linear layers.
        Parameters:
            init_fn (callable): A function to initialize the weights of layers.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Apply custom initialization to weights
                init_fn(module.weight)
                if module.bias is not None:
                    # Initialize biases uniformly
                    nn.init.uniform_(module.bias, -1.0, 1.0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Initialize BatchNorm layers
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

class PointNetDisClassifier(nn.Module):
    def __init__(self, feat_ind1, feat_ind2,
                 pointnet_ls_list, init_fn):
        super(PointNetDisClassifier, self).__init__()
        self.arity = 2
        self.feat_ind1 = feat_ind1
        self.feat_ind2 = feat_ind2
        self.feat = PointNetEncoder(pointnet_ls_list, \
                                    global_feat=True, \
                                    feature_transform=True)
        self.fc1 = nn.Linear(pointnet_ls_list[2], \
                             pointnet_ls_list[1])
        self.fc2 = nn.Linear(pointnet_ls_list[1], pointnet_ls_list[0])
        self.fc3 = nn.Linear(pointnet_ls_list[0], 1)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(pointnet_ls_list[1])
        self.bn2 = nn.BatchNorm1d(pointnet_ls_list[0])
        self.apply_init(init_fn)

    def forward(self, graph: Dict) -> List[Dict]:
        input_data = graph['edges']
        B = input_data.shape[0]
        input_data1 = input_data[:, self.feat_ind1]
        xyz1 = input_data1.view(B, -1, 3)
        input_data2 = input_data[:, self.feat_ind2]
        xyz2 = input_data2.view(B, -1, 3)
        # the distance between the two point clouds
        xyz = xyz1 - xyz2
        xyz = xyz.permute(0, 2, 1)  # xyz: B x 3 x N

        x = self.feat(xyz)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        decoded = self.fc3(x)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph

    def apply_init(self, init_fn):
        """
        Apply custom initialization to all submodules, including encoders and linear layers.
        Parameters:
            init_fn (callable): A function to initialize the weights of layers.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Apply custom initialization to weights
                init_fn(module.weight)
                if module.bias is not None:
                    # Initialize biases uniformly
                    nn.init.uniform_(module.bias, -1.0, 1.0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Initialize BatchNorm layers
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

class PointNetDisNormClassifier(nn.Module):
    def __init__(self, feat_pcd_ind1, feat_pcd_ind2,
                    feat_vec_ind1, feat_vec_ind2,
                 pointnet_ls_list, init_fn):
        super(PointNetDisNormClassifier, self).__init__()
        self.arity = 2
        self.feat_pcd_ind1 = feat_pcd_ind1
        self.feat_pcd_ind2 = feat_pcd_ind2
        self.feat_vec_ind1 = feat_vec_ind1
        self.feat_vec_ind2 = feat_vec_ind2
        self.feat = PointNetEncoder(pointnet_ls_list, \
                                    global_feat=True, \
                                    feature_transform=True, \
                                    channel=6)
        self.fc1 = nn.Linear(pointnet_ls_list[2], \
                             pointnet_ls_list[1])
        self.fc2 = nn.Linear(pointnet_ls_list[1], pointnet_ls_list[0])
        self.fc3 = nn.Linear(pointnet_ls_list[0], 1)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(pointnet_ls_list[1])
        self.bn2 = nn.BatchNorm1d(pointnet_ls_list[0])
        self.apply_init(init_fn)

    def forward(self, graph: Dict) -> List[Dict]:
        input_data = graph['edges']
        B = input_data.shape[0]
        input_pcd_data1 = input_data[:, self.feat_pcd_ind1]
        xyz1 = input_pcd_data1.view(B, -1, 3)
        input_pcd_data2 = input_data[:, self.feat_pcd_ind2]
        xyz2 = input_pcd_data2.view(B, -1, 3)
        # the distance between the two point clouds
        xyz = xyz1 - xyz2
        xyz = xyz.permute(0, 2, 1)  # xyz: B x 3 x N

        input_vec_data1 = input_data[:, self.feat_vec_ind1]
        vec1 = input_vec_data1.view(B, -1, 3)
        input_vec_data2 = input_data[:, self.feat_vec_ind2]
        vec2 = input_vec_data2.view(B, -1, 3)
        # the distance between the two point clouds norms
        vec = vec1 - vec2
        vec = vec.permute(0, 2, 1)  # xyz: B x 3 x N

        x = self.feat(torch.cat([xyz, vec], dim=1))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        decoded = self.fc3(x)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph

    def apply_init(self, init_fn):
        """
        Apply custom initialization to all submodules, including encoders and linear layers.
        Parameters:
            init_fn (callable): A function to initialize the weights of layers.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Apply custom initialization to weights
                init_fn(module.weight)
                if module.bias is not None:
                    # Initialize biases uniformly
                    nn.init.uniform_(module.bias, -1.0, 1.0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Initialize BatchNorm layers
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

class PointNetRelClassifier(nn.Module):
    def __init__(self, feat_ind1, feat_ind2,
                 pointnet_ls_list, init_fn):
        super(PointNetRelClassifier, self).__init__()
        self.arity = 2
        self.feat_ind1 = feat_ind1
        self.feat_ind2 = feat_ind2
        self.feat = PointNetEncoder(pointnet_ls_list, \
                                    global_feat=True, \
                                    feature_transform=True)
        self.fc1 = nn.Linear(pointnet_ls_list[2], \
                             pointnet_ls_list[1])
        self.fc2 = nn.Linear(pointnet_ls_list[1], pointnet_ls_list[0])
        self.fc3 = nn.Linear(pointnet_ls_list[0], 1)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(pointnet_ls_list[1])
        self.bn2 = nn.BatchNorm1d(pointnet_ls_list[0])
        self.apply_init(init_fn)

    def forward(self, graph: Dict) -> List[Dict]:
        input_data = graph['edges']
        B = input_data.shape[0]
        input_data1 = input_data[:, self.feat_ind1]
        xyz1 = input_data1.view(B, -1, 3)
        input_data2 = input_data[:, self.feat_ind2]
        xyz2 = input_data2.view(B, -1, 3)
        # concatenate the two point clouds as if they are from the same object
        xyz = torch.cat([xyz1, xyz2], dim=1)
        xyz = xyz.permute(0, 2, 1)  # xyz: B x 3 x N

        x = self.feat(xyz)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        decoded = self.fc3(x)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph

    def apply_init(self, init_fn):
        """
        Apply custom initialization to all submodules, including encoders and linear layers.
        Parameters:
            init_fn (callable): A function to initialize the weights of layers.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Apply custom initialization to weights
                init_fn(module.weight)
                if module.bias is not None:
                    # Initialize biases uniformly
                    nn.init.uniform_(module.bias, -1.0, 1.0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Initialize BatchNorm layers
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

class PointNetClassifier(nn.Module):
    def __init__(self, feat_ind, pointnet_ls_list, init_fn):
        super(PointNetClassifier, self).__init__()
        self.feat_ind = feat_ind
        self.arity = 1
        self.feat = PointNetEncoder(pointnet_ls_list, \
                                    global_feat=True, \
                                    feature_transform=True)
        self.fc1 = nn.Linear(pointnet_ls_list[2], \
                             pointnet_ls_list[1])
        self.fc2 = nn.Linear(pointnet_ls_list[1], pointnet_ls_list[0])
        self.fc3 = nn.Linear(pointnet_ls_list[0], 1)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(pointnet_ls_list[1])
        self.bn2 = nn.BatchNorm1d(pointnet_ls_list[0])
        self.apply_init(init_fn)

    def forward(self, graph: Dict) -> List[Dict]:
        input_data = graph['nodes']
        B = input_data.shape[0]
        input_data = input_data[:, self.feat_ind]
        xyz = input_data.view(B, -1, 3)
        xyz = xyz.permute(0, 2, 1)  # xyz: B x 3 x N

        x = self.feat(xyz)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        decoded = self.fc3(x)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph

    def apply_init(self, init_fn):
        """
        Apply custom initialization to all submodules, including encoders and linear layers.
        Parameters:
            init_fn (callable): A function to initialize the weights of layers.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Apply custom initialization to weights
                init_fn(module.weight)
                if module.bias is not None:
                    # Initialize biases uniformly
                    nn.init.uniform_(module.bias, -1.0, 1.0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Initialize BatchNorm layers
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.activation(out)
        out = self.norm(out)
        if x.shape[-1] != out.shape[-1]:
            identity = nn.functional.pad(identity, (0, out.shape[-1] - x.shape[-1]), "constant", 0)
        out += identity
        return out

def MLP(layers: List[int], input_dim: int, with_ln: bool) -> nn.Sequential:
    """Create MLP."""
    LinearLayer = nn.Linear
    mlp_layers: List[nn.Module] = [LinearLayer(input_dim, layers[0])]

    for layer_num in range(0, len(layers) - 1):
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.BatchNorm1d(layers[layer_num]))
        # mlp_layers.append(nn.Dropout(p=0.5))
        mlp_layers.append(LinearLayer(layers[layer_num],
                                      layers[layer_num + 1]))

    # Add a final normalization layer
    if with_ln:
        mlp_layers.append(nn.LayerNorm(layers[-1]))  # type: ignore

    return nn.Sequential(*mlp_layers)

class EncodeDecode(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, dims: List[int],
                 **kwargs: List[int]) -> None:
        super().__init__()
        # dims = [node_dim, edge_dim, global_dim, node_dim, edge_dim, global_dim]
        # input dims should > 0, otherwise will have ones padding 
        node_dim = dims[0] if dims[0] > 0 else 0
        edge_dim = dims[1] if dims[1] > 0 else 0
        global_dim = dims[2] if dims[2] > 0 else 0
        self.params = []

        if 'node_encoder_layers' in kwargs:
            self.node_encoder = MLP(kwargs['node_encoder_layers'], node_dim, True)
            node_dim = kwargs['node_encoder_layers'][-1]
            self.params.append(self.node_encoder.parameters())
        if 'edge_encoder_layers' in kwargs:
            self.edge_encoder = MLP(kwargs['edge_encoder_layers'], edge_dim, True)
            edge_dim = kwargs['edge_encoder_layers'][-1]
            self.params.append(self.edge_encoder.parameters())
        if 'global_encoder_layers' in kwargs:
            self.global_encoder = MLP(kwargs['global_encoder_layers'],
                                      global_dim, True)
            global_dim = kwargs['global_encoder_layers'][-1]
            self.params.append(self.global_encoder.parameters())
        # no message passing, this is actually just an MLP
        if 'node_decoder_layers' in kwargs:
            self.node_decoder = MLP(kwargs['node_decoder_layers'],
                                    node_dim, False)
        if 'edge_decoder_layers' in kwargs:
            self.edge_decoder = MLP(kwargs['edge_decoder_layers'],
                                    edge_dim, False)
        if 'global_decoder_layers' in kwargs:
            self.global_decoder = MLP(kwargs['global_decoder_layers'],
                                      global_dim, False)
            
    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        if hasattr(self, 'node_encoder'):
            data = graph['nodes']
            if data.size()[1] == 0:
                data = torch.ones((data.size()[0], 1)).to(data.device)
            graph['nodes'] = self.node_encoder(data)
        if hasattr(self, 'edge_encoder'):
            data = graph['edges']
            if data.size()[1] == 0:
                data = torch.ones((data.size()[0], 1)).to(data.device)
            graph['edges'] = self.edge_encoder(data)
        if hasattr(self, 'global_encoder'):
            data = graph['globals']
            if data.size()[1] == 0:
                data = torch.ones((data.size()[0], 1)).to(data.device)
            graph['globals'] = self.global_encoder(data)
        else:
            graph['globals'] = torch.zeros((graph['n_node'].size()[0], 0)).to(graph['nodes'].device)

        output_graph = []

        replacements = {
            'nodes': self.node_decoder(graph['nodes']),
            'edges': self.edge_decoder(graph['edges']),
        }

        if hasattr(self, 'global_decoder'):
            replacements['globals'] = self.global_decoder(graph['globals'])
        else:
            replacements['globals'] = graph['globals']

        output_graph.append(replace_graph(graph, replacements))

        return output_graph
    
    def reset_parameters(self):
        for param in self.params:
            for p in param:
                if len(p.shape) > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)

class EncodeDecodeMLP(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, arity: int, \
                 encoder: nn.Module, \
                 decoder: nn.Module, \
                 init_fn) -> None:
        super().__init__()
        # dims = [node_dim, edge_dim, global_dim, node_dim, edge_dim, global_dim]
        # input dims should > 0, otherwise will have ones padding 
        self.arity = arity
        self.encoder = encoder
        self.decoder = decoder
        self.apply_init(init_fn)

    def apply_init(self, init_fn):
        """Apply custom initialization to the encoder and decoder."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                # Bias use uniform initialization
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        if self.arity == 1:
            input_data = graph['nodes']
        elif self.arity == 2:
            input_data = graph['edges']

        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph
    
class SelectiveRotEncodeDecodeMLP(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, arity: int, \
                 encoder: nn.Module, \
                 encoder_rot: nn.Module, \
                 decoder: nn.Module, \
                 non_rot_feat: List[int], \
                 rot_feat: Dict, \
                 init_fn) -> None:
        super().__init__()
        # dims = [node_dim, edge_dim, global_dim, node_dim, edge_dim, global_dim]
        # input dims should > 0, otherwise will have ones padding 
        self.non_rot_feat_inx = non_rot_feat
        self.rot_feat_ind = []
        for i in range(1, arity+1):
            local_rot = [None] * 4
            for feat in rot_feat.keys():
                # note that real part is the first element
                if "ent{}".format(i) in feat and "_qw" in feat:
                    local_rot[0] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qx" in feat:
                    local_rot[1] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qy" in feat:
                    local_rot[2] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qz" in feat:
                    local_rot[3] = rot_feat[feat]
            self.rot_feat_ind.append(local_rot)
        self.arity = arity
        self.encoder = encoder
        self.encoder_rot = encoder_rot
        self.decoder = decoder
        self.apply_init(init_fn)

    def apply_init(self, init_fn):
        """Apply custom initialization to the encoder and decoder."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                # Bias use uniform initialization
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        if self.arity == 1:
            input_data = graph['nodes']
            input_rot_quant = input_data[:, self.rot_feat_ind[0]]
            input_rot = quaternion_to_matrix(input_rot_quant).view(-1, 9)
        elif self.arity == 2:
            input_data = graph['edges']
            input_rot_quant_1 = input_data[:, self.rot_feat_ind[0]]
            if not (torch.isclose(torch.norm(input_rot_quant_1, dim=1), \
                    torch.ones_like(torch.norm(input_rot_quant_1, dim=1))).all()):
                logging.warning("Input quaternion does not have unit norm")
            input_rot_1 = quaternion_to_matrix(input_rot_quant_1).view(-1, 9)
            input_rot_quant_2 = input_data[:, self.rot_feat_ind[1]]
            if not torch.isclose(torch.norm(input_rot_quant_2, dim=1), \
                    torch.ones_like(torch.norm(input_rot_quant_2, dim=1))).all():
                logging.warning("Input quaternion does not have unit norm")
            input_rot_2 = quaternion_to_matrix(input_rot_quant_2).view(-1, 9)
            input_rot = torch.cat([input_rot_1, input_rot_2], dim=1)

        input_feat = input_data[:, self.non_rot_feat_inx]
        encoded_non_rot = self.encoder(input_feat)
        encoded_rot = self.encoder_rot(input_rot)
        encoded = torch.cat([encoded_non_rot, encoded_rot], dim=1)
        decoded = self.decoder(encoded)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph
    
class SelectiveEncodeDecodeMLP(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, arity: int, \
                 encoder: nn.Module, \
                 decoder: nn.Module, \
                 feat_ind: List[int], \
                 init_fn) -> None:
        super().__init__()
        # dims = [node_dim, edge_dim, global_dim, node_dim, edge_dim, global_dim]
        # input dims should > 0, otherwise will have ones padding 
        self.feat_ind = feat_ind
        self.arity = arity
        assert self.arity == 1, "Only support arity == 1 for selective feature"
        self.encoder = encoder
        self.decoder = decoder
        self.apply_init(init_fn)

    def apply_init(self, init_fn):
        """Apply custom initialization to the encoder and decoder."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                # Bias use uniform initialization
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        input_data = graph['nodes']
        input_feat = input_data[:, self.feat_ind]
        encoded = self.encoder(input_feat)
        decoded = self.decoder(encoded)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph
    
class SelectiveRelRotEncodeDecodeMLP(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, arity: int, \
                 encoder: nn.Module, \
                 encoder_rot: nn.Module, \
                 decoder: nn.Module, \
                 non_rot_feat: Dict, \
                 rot_feat: Dict, \
                 init_fn) -> None:
        super().__init__()
        # dims = [node_dim, edge_dim, global_dim, node_dim, edge_dim, global_dim]
        # input dims should > 0, otherwise will have ones padding 
        assert arity == 2, "Only support arity == 2 for relative feature"
        self.non_rot_feat_inx = []
        self.rot_feat_ind = []
        for i in range(1, arity+1):
            local_rot = [None] * 4
            for feat in rot_feat.keys():
                # note that real part is the first element
                if "ent{}".format(i) in feat and "_qw" in feat:
                    local_rot[0] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qx" in feat:
                    local_rot[1] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qy" in feat:
                    local_rot[2] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qz" in feat:
                    local_rot[3] = rot_feat[feat]
            self.rot_feat_ind.append(local_rot)
            # non rot also compute relative
            local_non_root = []
            for feat in non_rot_feat:
                if "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
            self.non_rot_feat_inx.append(local_non_root)
        self.arity = arity
        self.encoder = encoder
        self.encoder_rot = encoder_rot
        self.decoder = decoder
        self.apply_init(init_fn)

    def apply_init(self, init_fn):
        """Apply custom initialization to the encoder and decoder."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                # Bias use uniform initialization
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        input_data = graph['edges']
        input_rot_quant_1 = input_data[:, self.rot_feat_ind[0]]
        assert torch.isclose(torch.norm(input_rot_quant_1, dim=1), \
                torch.ones_like(torch.norm(input_rot_quant_1, dim=1))).all()
        input_rot_1 = quaternion_to_matrix(input_rot_quant_1).view(-1, 9)
        input_rot_quant_2 = input_data[:, self.rot_feat_ind[1]]
        assert torch.isclose(torch.norm(input_rot_quant_2, dim=1), \
                torch.ones_like(torch.norm(input_rot_quant_2, dim=1))).all()
        input_rot_2 = quaternion_to_matrix(input_rot_quant_2).view(-1, 9)
        # additionally compute the relative rotation
        rel_rot = torch.matmul(input_rot_1.view(-1, 3, 3), input_rot_2.view(-1, 3, 3).transpose(1, 2))
        rel_rot = rel_rot.view(-1, 9)
        input_rot = torch.cat([input_rot_1, input_rot_2, rel_rot], dim=1)

        input_non_rot_1 = input_data[:, self.non_rot_feat_inx[0]]
        input_non_rot_2 = input_data[:, self.non_rot_feat_inx[1]]
        rel_non_rot = input_non_rot_1 - input_non_rot_2
        input_feat = torch.cat([input_non_rot_1, input_non_rot_2, rel_non_rot], dim=1)
        encoded_non_rot = self.encoder(input_feat)
        encoded_rot = self.encoder_rot(input_rot)
        encoded = torch.cat([encoded_non_rot, encoded_rot], dim=1)
        decoded = self.decoder(encoded)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph

class SelectiveRelOnlyEncodeDecodeMLP(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, arity: int, \
                 encoder: nn.Module, \
                 decoder: nn.Module, \
                 non_rot_feat: Dict, \
                 rot_feat: Dict, \
                 init_fn) -> None:
        super().__init__()
        # dims = [node_dim, edge_dim, global_dim, node_dim, edge_dim, global_dim]
        # input dims should > 0, otherwise will have ones padding 
        assert arity == 2, "Only support arity == 2 for relative feature"
        self.non_rot_feat_inx = []
        self.rot_feat_ind = []
        for i in range(1, arity+1):
            local_rot = [None] * 4
            for feat in rot_feat.keys():
                # note that real part is the first element
                if "ent{}".format(i) in feat and "_qw" in feat:
                    local_rot[0] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qx" in feat:
                    local_rot[1] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qy" in feat:
                    local_rot[2] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qz" in feat:
                    local_rot[3] = rot_feat[feat]
            self.rot_feat_ind.append(local_rot)
            # non rot also compute relative
            local_non_root = []
            for feat in non_rot_feat:
                if "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
            assert len(local_non_root) == 3, "Only support 3 non-rot features (x,y,z)"
            self.non_rot_feat_inx.append(local_non_root)
        self.arity = arity
        self.encoder = encoder
        self.decoder = decoder
        self.apply_init(init_fn)

    def apply_init(self, init_fn):
        """Apply custom initialization to the encoder and decoder."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                # Bias use uniform initialization
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        input_data = graph['edges']
        input_rot_quant_1 = input_data[:, self.rot_feat_ind[0]]
        assert torch.isclose(torch.norm(input_rot_quant_1, dim=1), \
                torch.ones_like(torch.norm(input_rot_quant_1, dim=1))).all()
        input_rot_1 = quaternion_to_matrix(input_rot_quant_1).view(-1, 3, 3)
        input_rot_quant_2 = input_data[:, self.rot_feat_ind[1]]
        assert torch.isclose(torch.norm(input_rot_quant_2, dim=1), \
                torch.ones_like(torch.norm(input_rot_quant_2, dim=1))).all()
        input_rot_2 = quaternion_to_matrix(input_rot_quant_2).view(-1, 3, 3)
        # additionally compute the relative rotation

        input_trans_1 = input_data[:, self.non_rot_feat_inx[0]]
        input_trans_2 = input_data[:, self.non_rot_feat_inx[1]]
        
        input_pose_1 = torch.zeros((input_trans_1.shape[0], 4, 4)).to(input_trans_1.device)
        input_pose_1[:, :3, :3] = input_rot_1
        input_pose_1[:, :3, 3] = input_trans_1
        input_pose_1[:, 3, 3] = 1.0

        input_pose_2 = torch.zeros((input_trans_2.shape[0], 4, 4)).to(input_trans_2.device)
        input_pose_2[:, :3, :3] = input_rot_2
        input_pose_2[:, :3, 3] = input_trans_2
        input_pose_2[:, 3, 3] = 1.0

        # rel pose
        rel_pose = torch.matmul(input_pose_1, input_pose_2.inverse())
        rel_pose = rel_pose.view(-1, 16)

        encoded = self.encoder(rel_pose)
        decoded = self.decoder(encoded)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph
    
class SelectiveRelPoseOnlyEncodeDecodeMLP(nn.Module, abc.ABC):
    """Encode-decode architecture."""

    def __init__(self, arity: int, \
                 encoder: nn.Module, \
                 encoder_rot: nn.Module, \
                 decoder: nn.Module, \
                 non_rot_feat: Dict, \
                 rot_feat: Dict, \
                 init_fn) -> None:
        super().__init__()
        # dims = [node_dim, edge_dim, global_dim, node_dim, edge_dim, global_dim]
        # input dims should > 0, otherwise will have ones padding 
        assert arity == 2, "Only support arity == 2 for relative feature"
        self.non_rot_feat_inx = []
        self.rot_feat_ind = []
        for i in range(1, arity+1):
            local_rot = [None] * 4
            for feat in rot_feat.keys():
                # note that real part is the first element
                if "ent{}".format(i) in feat and "_qw" in feat:
                    local_rot[0] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qx" in feat:
                    local_rot[1] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qy" in feat:
                    local_rot[2] = rot_feat[feat]
                elif "ent{}".format(i) in feat and "_qz" in feat:
                    local_rot[3] = rot_feat[feat]
            self.rot_feat_ind.append(local_rot)
            # non rot also compute relative
            local_non_root = []
            for feat in non_rot_feat:
                if "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
                elif "ent{}".format(i) in feat:
                    local_non_root.append(non_rot_feat[feat])
            self.non_rot_feat_inx.append(local_non_root)
        self.arity = arity
        self.encoder = encoder
        self.encoder_rot = encoder_rot
        self.decoder = decoder
        self.apply_init(init_fn)

    def apply_init(self, init_fn):
        """Apply custom initialization to the encoder and decoder."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                # Bias use uniform initialization
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        input_data = graph['edges']
        input_rot_quant_1 = input_data[:, self.rot_feat_ind[0]]
        assert torch.isclose(torch.norm(input_rot_quant_1, dim=1), \
                torch.ones_like(torch.norm(input_rot_quant_1, dim=1))).all()
        input_rot_1 = quaternion_to_matrix(input_rot_quant_1).view(-1, 9)
        input_rot_quant_2 = input_data[:, self.rot_feat_ind[1]]
        assert torch.isclose(torch.norm(input_rot_quant_2, dim=1), \
                torch.ones_like(torch.norm(input_rot_quant_2, dim=1))).all()
        input_rot_2 = quaternion_to_matrix(input_rot_quant_2).view(-1, 9)
        # additionally compute the relative rotation
        rel_rot = torch.matmul(input_rot_1.view(-1, 3, 3), input_rot_2.view(-1, 3, 3).transpose(1, 2))
        rel_rot = rel_rot.view(-1, 9)

        input_non_rot_1 = input_data[:, self.non_rot_feat_inx[0]]
        input_non_rot_2 = input_data[:, self.non_rot_feat_inx[1]]
        rel_non_rot = input_non_rot_1 - input_non_rot_2

        encoded_non_rot = self.encoder(rel_non_rot)
        encoded_rot = self.encoder_rot(rel_rot)
        encoded = torch.cat([encoded_non_rot, encoded_rot], dim=1)
        decoded = self.decoder(encoded)

        output_graph = []
        if self.arity == 1:
            replacements = {
                'nodes': decoded
            }
        elif self.arity == 2:
            replacements = {
                'edges': decoded
            }
        output_graph.append(replace_graph(graph, replacements))

        return output_graph
    
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

def setup_neupi_mlp_net(graph_dataset: GraphTransC2DDataset, 
                    pred_arity: int,
                    archi: Dict,
                    node_feat2inx: Dict,
                    edge_feat2inx: Dict,
                    dims: List[int] = None
                    ) -> EncodeDecodeMLP:
    """Create an EncodeProcessDecode GNN using the dimensions found in the
    dataset."""
    if dims is None:
        dims = _setup_neupi_dims(graph_dataset)
    else:
        assert graph_dataset is None, "dims and graph_dataset cannot be both provided"
    if pred_arity == 1:
        input_dim = dims[0]
    elif pred_arity == 2:
        input_dim = dims[1]
    else:
        raise ValueError(f"Unsupported pred_arity: {pred_arity}")
    if archi["type"] == "MLP":
        assert "layer_size" in archi
        layer_size = archi["layer_size"]
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*2, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_mlp = EncodeDecodeMLP(pred_arity, encoder_model, decoder_model, init_fn)
        return connected_mlp
    elif archi["type"] == "Selective-PosRot-MLP":
        assert "feature_names" in archi
        assert "layer_size" in archi
        layer_size = archi["layer_size"]
        rot_ind = {}
        non_rot_ind = []
        if pred_arity == 1:
            for feat in node_feat2inx.keys():
                if feat in archi["feature_names"]:
                    if '_q' in feat:
                        rot_ind[feat] = node_feat2inx[feat]
                    else:
                        non_rot_ind.append(node_feat2inx[feat])
        elif pred_arity == 2:
            for feat in edge_feat2inx.keys():
                if feat in archi["feature_names"]:
                    if '_q' in feat:
                        rot_ind[feat] = edge_feat2inx[feat]
                    else:
                        non_rot_ind.append(edge_feat2inx[feat])
        input_dim = len(non_rot_ind)
        rot_input_dim = 9 if pred_arity == 1 else 18
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        encoder_model_rot = MLP([layer_size, layer_size*2], rot_input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*4, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_selective_mlp = SelectiveRotEncodeDecodeMLP(pred_arity, 
                                encoder_model, encoder_model_rot,
                                decoder_model, non_rot_ind, rot_ind, init_fn)
        return connected_selective_mlp
    elif archi["type"] == "Selective-PosRelRot-MLP":
        assert "feature_names" in archi
        assert "layer_size" in archi
        assert pred_arity == 2, "Only support pred_arity == 2 for relative feature"
        layer_size = archi["layer_size"]
        rot_ind = {}
        non_rot_ind = {}
        for feat in edge_feat2inx.keys():
            if feat in archi["feature_names"]:
                if '_q' in feat:
                    rot_ind[feat] = edge_feat2inx[feat]
                else:
                    non_rot_ind[feat] = edge_feat2inx[feat]
        assert len(non_rot_ind) % 2 == 0, "Only support even number of non-rot features"
        input_dim = len(non_rot_ind) * 3 // 2 # additional 0.5 for relative rot
        rot_input_dim = 27 # additional 9 for relative rot
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        encoder_model_rot = MLP([layer_size, layer_size*2], rot_input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*4, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_selective_mlp = SelectiveRelRotEncodeDecodeMLP(pred_arity, 
                                encoder_model, encoder_model_rot,
                                decoder_model, non_rot_ind, rot_ind, init_fn)
        return connected_selective_mlp
    elif archi["type"] == "Selective-RelOnly-MLP":
        assert "feature_names" in archi
        assert "layer_size" in archi
        assert pred_arity == 2, "Only support pred_arity == 2 for relative feature"
        layer_size = archi["layer_size"]
        rot_ind = {}
        non_rot_ind = {}
        for feat in edge_feat2inx.keys():
            if feat in archi["feature_names"]:
                if '_q' in feat:
                    rot_ind[feat] = edge_feat2inx[feat]
                else:
                    non_rot_ind[feat] = edge_feat2inx[feat]
        assert len(non_rot_ind) % 2 == 0, "Only support even number of non-rot features"
        input_dim = len(non_rot_ind) // 2 # 0.5 for relative pose
        rot_input_dim = 9 # 9 for relative rot
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        encoder_model_rot = MLP([layer_size, layer_size*2], rot_input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*4, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_selective_mlp = SelectiveRelOnlyEncodeDecodeMLP(pred_arity, 
                                encoder_model, encoder_model_rot,
                                decoder_model, non_rot_ind, rot_ind, init_fn)
        return connected_selective_mlp
    elif archi["type"] == "Selective-RelPoseOnly-MLP":
        assert "feature_names" in archi
        assert "layer_size" in archi
        assert pred_arity == 2, "Only support pred_arity == 2 for relative feature"
        layer_size = archi["layer_size"]
        rot_ind = {}
        non_rot_ind = {}
        for feat in edge_feat2inx.keys():
            if feat in archi["feature_names"]:
                if '_q' in feat:
                    rot_ind[feat] = edge_feat2inx[feat]
                else:
                    non_rot_ind[feat] = edge_feat2inx[feat]
        assert len(non_rot_ind) % 2 == 0, "Only support even number of non-rot features"
        input_dim = 16 # 0.5 for relative pose
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*2, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_selective_mlp = SelectiveRelOnlyEncodeDecodeMLP(pred_arity, 
                                encoder_model, decoder_model, non_rot_ind, rot_ind, init_fn)
        return connected_selective_mlp
    elif archi["type"] == "Selective-MLP":
        assert "feature_names" in archi
        assert "layer_size" in archi
        layer_size = archi["layer_size"]
        feat_ind = []
        assert pred_arity == 1, "Only support pred_arity == 1 for selectiveMLP"
        for feat in node_feat2inx.keys():
            if feat in archi["feature_names"]:
                feat_ind.append(node_feat2inx[feat])
        input_dim = len(feat_ind)
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        decoder_model = MLP([layer_size*2, layer_size, 1], layer_size*2, False)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        connected_selective_mlp = SelectiveEncodeDecodeMLP(pred_arity, 
                                encoder_model, decoder_model,
                                feat_ind, init_fn)
        return connected_selective_mlp
    elif archi["type"] == "Selective-PointNet":
        feat_ind = []
        assert pred_arity == 1, "Only support pred_arity == 1 for selectiveMLP"
        for feat in node_feat2inx.keys():
            if 'pcd' in feat:
                feat_ind.append(node_feat2inx[feat])
        assert feat_ind is not None, "Feature not found"
        # TODO: implement the pointnet
        assert len(feat_ind) % 3 == 0, "Only support 3D point cloud"
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        pointnet_feat_lst = archi["pointnet_feat_lst"]
        classifier = PointNetClassifier(feat_ind, pointnet_feat_lst, init_fn)
        return classifier
    elif archi["type"] == "Selective-RelPointNet":
        feat_ind_1 = []
        feat_ind_2 = []
        assert pred_arity == 2, "Only support pred_arity == 2 for selectiveMLP"
        for feat in edge_feat2inx.keys():
            if 'pcd' in feat:
                if 'ent1' in feat:
                    feat_ind_1.append(edge_feat2inx[feat])
                elif 'ent2' in feat:
                    feat_ind_2.append(edge_feat2inx[feat])
        assert feat_ind_1 is not None, "Feature not found"
        assert feat_ind_2 is not None, "Feature not found"
        assert len(feat_ind_1) % 3 == 0, "Only support 3D point cloud"
        assert len(feat_ind_2) % 3 == 0, "Only support 3D point cloud"
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        pointnet_feat_lst = archi["pointnet_feat_lst"]
        classifier = PointNetRelClassifier(feat_ind_1, feat_ind_2, pointnet_feat_lst, init_fn)
        return classifier
    elif archi["type"] == "Selective-DisPointNet":
        feat_ind_1 = []
        feat_ind_2 = []
        assert pred_arity == 2, "Only support pred_arity == 2 for selectiveMLP"
        for feat in edge_feat2inx.keys():
            if 'pcd' in feat:
                if 'ent1' in feat:
                    feat_ind_1.append(edge_feat2inx[feat])
                elif 'ent2' in feat:
                    feat_ind_2.append(edge_feat2inx[feat])
        assert feat_ind_1 is not None, "Feature not found"
        assert feat_ind_2 is not None, "Feature not found"
        assert len(feat_ind_1) % 3 == 0, "Only support 3D point cloud"
        assert len(feat_ind_2) % 3 == 0, "Only support 3D point cloud"
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        pointnet_feat_lst = archi["pointnet_feat_lst"]
        classifier = PointNetDisClassifier(feat_ind_1, feat_ind_2, pointnet_feat_lst, init_fn)
        return classifier
    elif archi["type"] == "Selective-DisPointNormNet":
        feat_pcd_ind_1 = []
        feat_pcd_ind_2 = []
        feat_norm_ind_1 = []
        feat_norm_ind_2 = []
        assert pred_arity == 2, "Only support pred_arity == 2 for DisPointNormNet"
        for feat in edge_feat2inx.keys():
            if 'pcd' in feat:
                if 'ent1' in feat:
                    feat_pcd_ind_1.append(edge_feat2inx[feat])
                elif 'ent2' in feat:
                    feat_pcd_ind_2.append(edge_feat2inx[feat])
            elif 'norm' in feat:
                if 'ent1' in feat:
                    feat_norm_ind_1.append(edge_feat2inx[feat])
                elif 'ent2' in feat:
                    feat_norm_ind_2.append(edge_feat2inx[feat])
        assert len(feat_pcd_ind_1) % 3 == 0, "Only support 3D point cloud"
        assert len(feat_norm_ind_2) % 3 == 0, "Only support 3D point cloud"
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        pointnet_feat_lst = archi["pointnet_feat_lst"]
        classifier = PointNetDisNormClassifier(feat_pcd_ind_1, 
                                               feat_pcd_ind_2, 
                                                feat_norm_ind_1,
                                                feat_norm_ind_2,
                                               pointnet_feat_lst, init_fn)
        return classifier
    elif archi["type"] == "Selective-RelMLPPointNet":
        feat_ind = [[], []]
        assert pred_arity == 2, "Only support pred_arity == 2 for selectiveMLP"
        assert "pcd_ent" in archi, "Need to specify the pcd_ent"
        feat_names = archi["feature_names"]
        layer_size = archi["layer_size"]
        for feat in edge_feat2inx.keys():
            if archi["pcd_ent"] == 1:
                if ('pcd' in feat) and ('ent1' in feat):
                    feat_ind[0].append(edge_feat2inx[feat])
                elif feat in feat_names:
                    feat_ind[1].append(edge_feat2inx[feat])
            elif archi["pcd_ent"] == 2:
                if ('pcd' in feat) and ('ent2' in feat):
                    feat_ind[0].append(edge_feat2inx[feat])
                elif feat in feat_names:
                    feat_ind[1].append(edge_feat2inx[feat])
            else:
                raise ValueError(f"Unsupported pcd_ent: {archi['pcd_ent']}")
        assert feat_ind is not None, "Feature not found"
        assert len(feat_ind[0]) % 3 == 0, "Only support 3D point cloud"
        input_dim = len(feat_ind[1])
        encoder_model = MLP([layer_size, layer_size*2], input_dim, True)
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        pointnet_feat_lst = archi["pointnet_feat_lst"]
        classifier = PointNetMLPClassifier(feat_ind[0],
                    feat_ind[1], encoder_model,
                    layer_size*2, pointnet_feat_lst, init_fn)
        return classifier
    elif archi["type"] == "Selective-NormNet":
        feat_ind = []
        assert pred_arity == 1, "Only support pred_arity == 1 for selectiveMLP"
        for feat in node_feat2inx.keys():
            # only use the norm feature
            if 'norm' in feat:
                feat_ind.append(node_feat2inx[feat])
        assert feat_ind is not None, "Feature not found"
        # TODO: implement the pointnet
        assert len(feat_ind) % 3 == 0, "Only support 3D point cloud"
        # created a connect mlp with encoder and decoder names
        if 'initializer' in archi:
            init_fn = get_init_fn(archi['initializer'])
        else:
            logging.warning("No initializer specified, using default (kaiming).")
            init_fn = get_init_fn('kaiming')
        pointnet_feat_lst = archi["pointnet_feat_lst"]
        classifier = PointNetClassifier(feat_ind, pointnet_feat_lst, init_fn)
        return classifier
    else:
        raise ValueError(f"Unsupported architecture type: {archi['type']}")

def get_init_fn(init_type):
    """Function to retrieve different initialization functions."""
    if init_type == 'xavier':
        return nn.init.xavier_uniform_
    elif init_type == 'kaiming':
        return nn.init.kaiming_uniform_
    elif init_type == 'uniform':
        return nn.init.uniform_
    else:
        raise ValueError(f"Unknown initializer: {init_type}")
    

def setup_neupi_optimizer(model: nn.Module, opti_config: Dict, lr_scheduler_config: Dict = None) \
    -> torch.optim.Optimizer:
    # Choose the optimizer
    if opti_config['type'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), \
                                     **opti_config['kwargs'])
    elif opti_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), \
                                      **opti_config['kwargs'])
    else:
        raise ValueError(f"Unsupported optimizer type: {opti_config['type']}")
    
    # Learning rate scheduler setup
    if lr_scheduler_config is not None:
        if lr_scheduler_config['type'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                **lr_scheduler_config['kwargs'])
        elif lr_scheduler_config['type'] == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, \
                                **lr_scheduler_config['kwargs'])
        elif lr_scheduler_config['type'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, \
                                **lr_scheduler_config['kwargs'])
        elif lr_scheduler_config['type'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  \
                                **lr_scheduler_config['kwargs'])
        else:
            raise ValueError(f"Unsupported scheduler type: {lr_scheduler_config['type']}")
    else:
        scheduler = None
    
    return optimizer, scheduler

class DummyModel(torch.nn.Module):
    def __init__(self, random):
        super(DummyModel, self).__init__()
        self.place_holder_mlp = nn.Linear(1, 1)
        self.random = random

    def forward(self, x):
        if not self.random:
            return x
        else:
            random_graph = {
                "n_node": x["n_node"],
                "n_edge": x["n_edge"],
                "nodes": torch.zeros_like(x["nodes"]),
                "edges": torch.zeros_like(x["edges"]),
                "senders": x["senders"],
                "receivers": x["receivers"],
            }
            # rand the nodes as binary
            random_graph["nodes"] = torch.bernoulli(torch.rand_like(random_graph["nodes"]))
            # rand the edges as binary
            random_graph["edges"] = torch.bernoulli(torch.rand_like(random_graph["edges"]))
            # check if the "edges" are all zero
            for i in range(random_graph["edges"].shape[0]):
                if torch.sum(random_graph["edges"][i]) == 0:
                    random_graph["edges"][i, 0] = 1
            return random_graph
        
class DummyModelIndex(torch.nn.Module):
    def __init__(self, random, arity, idx):
        super(DummyModelIndex, self).__init__()
        self.place_holder_mlp = nn.Linear(1, 1)
        self.random = random
        self.arity = arity
        self.idx = idx

    def forward(self, x):
        if not self.random:
            if self.arity == 1:
                node_data = x["nodes"]
                node_data = node_data[:, self.idx:self.idx+1]
                new_graph = {
                    "n_node": x["n_node"],
                    "n_edge": x["n_edge"],
                    "nodes": node_data,
                    "edges": x["edges"],
                    "senders": x["senders"],
                    "receivers": x["receivers"],
                }
            elif self.arity == 2:
                edge_data = x["edges"]
                edge_data = edge_data[:, self.idx:self.idx+1]
                new_graph = {
                    "n_node": x["n_node"],
                    "n_edge": x["n_edge"],
                    "nodes": x["nodes"],
                    "edges": edge_data,
                    "senders": x["senders"],
                    "receivers": x["receivers"],
                }
            return new_graph
        else:
            if self.arity == 1:
                random_graph = {
                    "n_node": x["n_node"],
                    "n_edge": x["n_edge"],
                    "nodes": torch.zeros_like(x["nodes"])[:, 0:1],
                    "edges": x["edges"],
                    "senders": x["senders"],
                    "receivers": x["receivers"],
                }
                random_graph["nodes"] = torch.bernoulli(torch.rand_like(random_graph["nodes"]))
            elif self.arity == 2:
                random_graph = {
                    "n_node": x["n_node"],
                    "n_edge": x["n_edge"],
                    "nodes": x["nodes"],
                    "edges": torch.zeros_like(x["edges"])[:, 0:1],
                    "senders": x["senders"],
                    "receivers": x["receivers"],
                }
                random_graph["edges"] = torch.bernoulli(torch.rand_like(random_graph["edges"]))
            return random_graph

class MCTSNode:
    def __init__(self, state, level, parent=None):
        self.state = state          # The game state at this node
        self.parent = parent        # Parent node
        self.level = level          # Level in the tree
        self.visits = 0             # Number of times node has been visited
        self.guidance = np.zeros_like(state) - 1  # Guidance values init
        self.value = 0       # Total reward for this node
        self.next_states = set()  # Set of next states to avoid duplicates
        for i in range(len(self.state)):
            # next state only flip one "0" to "1 or "2"
            if self.state[i] == 0:
                for new_val in [1, 2]:  # Only allow 1 or 2
                    next_state = self.state.copy()
                    next_state[i] = new_val
                    self.next_states.add(tuple(next_state))  # Store as tuple for set

    def is_fully_expanded(self):
        return len(self.next_states) == 0
    
    def update_value(self, zero_loss, thresh):
        non_zero_indexes = np.where(self.state != 0)[0]
        non_zero_guidance = self.guidance[non_zero_indexes]
        # note that maximum value is 0.5
        # do not skip root
        if (non_zero_guidance.mean() < 0.4) or (non_zero_indexes.size == 0):
            local_zero_loss = zero_loss.copy()
            # non-zero indexes
            local_zero_loss[non_zero_indexes] = 0.0
            # count how large the loss of "zero" cells are, larger the more likely to be selected
            self.value = local_zero_loss.mean()
            if self.guidance.mean() > thresh:
                # It is far from the threshold, then large value
                self.value += self.guidance.mean() - thresh
        else:
            # non-zero loss is too large, wrong element wise supervision
            self.value = -np.inf

    def expand(self, zero_loss):
        # Expand up to batch_size untried actions
        next_state = self.get_successor(zero_loss)
        return MCTSNode(next_state, level=self.level+1, parent=self)

    def get_successor(self, zero_loss):
        local_zero_loss = zero_loss.copy()
        # non-zero indexes
        non_zero_indexes = np.where(self.state != 0)[0]
        local_zero_loss[non_zero_indexes] = -1
        # try to find a max zero loss index that has not been tried
        while not (local_zero_loss==-1).all():
            assert not self.is_fully_expanded()
            max_zero_loss_idx = np.argmax(local_zero_loss)
            next_state = self.state.copy()
            next_state_ = self.state.copy()
            next_state[max_zero_loss_idx] = 1
            next_state_[max_zero_loss_idx] = 2
            if array_in_list(next_state, self.next_states):
                self.next_states.remove(tuple(next_state))
                # logging.info(f"Found next state: {next_state}")
                return next_state
            elif array_in_list(next_state_, self.next_states):
                self.next_states.remove(tuple(next_state_))
                # logging.info(f"Found next state: {next_state_}")
                return next_state_
            else:
                # this cell has been tried, set it to -1
                local_zero_loss[max_zero_loss_idx] = -1
        if not self.is_fully_expanded():
            # randomly select one
            list_next_states = list(self.next_states)
            list_idx = np.random.choice(len(list_next_states))
            next_state = np.array(list_next_states[list_idx])
            self.next_states.remove(tuple(next_state))
            return next_state
        else:
            raise ValueError("Node is fully expanded. Should not be considered")

class HierachicalMCTSearcher(abc.ABC):
    """Uniformally sample the vectors."""

    def __init__(self, 
                 dim_num: int, \
                 frontier_max_level: int, \
                 guidance_th: float) -> None:
        self.visits = 0  # Total visits for UCT calculation
        self.guidance_th = min(guidance_th * dim_num / 2, 0.5)
        self.frontier_max_level = frontier_max_level
        root = MCTSNode(np.zeros(dim_num, dtype=np.int32), 0)
        root.guidance = np.zeros(root.state.shape)
        self.global_zero_loss = np.zeros(root.state.shape)
        root.update_value(self.global_zero_loss, self.guidance_th)
        self.frontier = [root]
        self.evaluated_values = {tuple(root.state)}

    def uct_selection(self, nodes, bs):
        # Use NumPy for efficient computation of UCT values
        visits = np.array([node.visits for node in nodes])
        values = np.array([node.value for node in nodes])
        
        # Avoid division by zero by setting a small value for nodes with zero visits
        visits = np.where(visits == 0, 1e-5, visits)
        
        total_visits = np.log(self.visits + 1)
        
        # UCT calculation
        uct_values = values / visits + 14.1 * self.guidance_th * np.sqrt(total_visits / visits)
        
        # Select top bs nodes by UCT values
        order = np.argsort(uct_values)[-bs:][::-1]
        
        # Return the top bs nodes in descending order of UCT values
        selected_nodes = [nodes[i] for i in order]
        
        return selected_nodes
    
    def update_value(self, state, guidance):
        # Update zero loss
        if (guidance != np.inf).all():
            # satisfiable
            zero_mask = state == 0
            self.global_zero_loss[zero_mask] += guidance[zero_mask]
            for node in self.frontier:
                if (node.state == state).all():
                    node.guidance = guidance.copy()
                node.update_value(self.global_zero_loss / self.visits, self.guidance_th)
        else:
            # unsatisfiable
            for node in self.frontier:
                if (node.state == state).all():
                    node.guidance = np.ones_like(node.guidance)
                    node.update_value(self.global_zero_loss / self.visits, self.guidance_th)
        
        self.update_front()

    def update_front(self):
        new_frontier = []
        for node in self.frontier:
            if (not node.is_fully_expanded()) and (node.value > -np.inf):
                new_frontier.append(node)
            else:
                logging.info(f"Node {node.state} is fully expanded or has -inf value.")
        if len(new_frontier) != len(self.frontier):
            logging.info(f"New Frontier: {[node.state for node in new_frontier]}")
        self.frontier = new_frontier
    
    def propose(self):
        # propose the next node to evaluate
        self.update_front()
        if len(self.frontier) == 0:
            return None
        else:
            # 1. Select a parent node using UCT
            while True:
                self.update_front()
                if self.frontier == []:
                    return None
                self.visits += 1
                selected_nodes = self.uct_selection(self.frontier, 1)
                # logging.info(f"Selected nodes: {[node.state for node in selected_nodes]}")
                # 2. Expand the selected nodes, add to frontier, and evaluate
                for node in selected_nodes:
                    assert not node.is_fully_expanded()
                    assert node.value > -np.inf
                    node.visits += 1
                    child = node.expand(self.global_zero_loss)
                    if tuple(child.state) not in self.evaluated_values:
                        self.evaluated_values.add(tuple(child.state))
                        # logging.info(f"Propose Child node: {child.state}")
                        # logging.info(f"From Parent: {node.state}")
                        if child.level <= self.frontier_max_level:
                            # logging.info(f"Adding to Frontier")
                            self.frontier.append(child)
                        return child.state