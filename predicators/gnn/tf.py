import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from .gnn import GraphModel, MLP, _setup_dims, GraphDictDataset, \
    _setup_dims_points
from predicators.gnn.pointnet_utils import PointNetEncoder
##############
# TRANSFORMER FOR GRAPHS
##############

class TransformerGraphModel(GraphModel):
    """
    A Transformer-based model that:
      1) Encodes node, edge, and global features (via optional MLPs).
      2) Flattens them into tokens, pads them, builds attention masks.
      3) Processes them with a Transformer.
      4) Unpads/unflattens the outputs back into node, edge, global features.
      5) Decodes (optional).
    """

    def __init__(self, 
                 dims: List[int],
                 d_model: int, 
                 nhead: int, 
                 num_tf_layers: int,
                 **kwargs):
        """
        :param dims: [node_input_dim, edge_input_dim, global_input_dim].
        :param d_model: Embedding dimension for the Transformer.
        :param nhead: Number of attention heads.
        :param num_layers: Number of TransformerEncoder layers.
        :param kwargs: Possibly includes:
            - node_encoder_layers, edge_encoder_layers, global_encoder_layers
            - node_decoder_layers, edge_decoder_layers, global_decoder_layers
            ... or other hyperparameters.
        """
        super().__init__(dims, **kwargs)
        if 'node_decoder_layers' in kwargs:
            self.node_decoder = MLP(kwargs['node_decoder_layers'],
                                    d_model)
        if 'edge_decoder_layers' in kwargs:
            self.edge_decoder = MLP(kwargs['edge_decoder_layers'],
                                    d_model)
        if 'global_decoder_layers' in kwargs:
            self.global_decoder = MLP(kwargs['global_decoder_layers'],
                                    d_model)

        # We'll project node_dim, edge_dim, global_dim to a single d_model 
        # so we can unify them as tokens in the Transformer
        self.node_project = nn.Linear(dims[0], d_model)
        self.edge_project = nn.Linear(dims[1], d_model)
        self.global_project = nn.Linear(dims[2], d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead,
                                                   dim_feedforward=4*d_model,
                                                   batch_first=True)  
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_tf_layers)

        # For convenience, store the dimension
        self.d_model = d_model

    def forward(self, graph: Dict) -> Dict:
        """
        Steps:
          1) MLP encode node, edge, global features (optional).
          2) Flatten them into tokens, pad, build attention mask.
          3) Run through Transformer.
          4) Unpad, unflatten to node/edge/global.
          5) MLP decode (optional).
          6) Return updated graph.
        """
        in_graph = graph.copy()
        # 1) Encode with MLP (if defined)
        if hasattr(self, 'node_encoder'):
            graph['nodes'] = self.node_encoder(graph['nodes'])  # shape: [#nodes, node_dim]
        if hasattr(self, 'edge_encoder'):
            graph['edges'] = self.edge_encoder(graph['edges'])  # shape: [#edges, edge_dim]
        if hasattr(self, 'global_encoder'):
            if graph['globals'] is not None:
                if graph['globals'].shape[1] == 0:
                    # handel empty globals
                    graph['globals'] = torch.ones_like(graph['n_node'])\
                        .to(graph['nodes'].dtype)
                graph['globals'] = self.global_encoder(graph['globals'])
            else:
                graph['globals'] = None

        # 2) Flatten nodes, edges, globals into tokens
        tokens, mask, metadata = self._flatten_and_pad(graph)  
        # 3) Pass through Transformer as message passing
        #    src_key_padding_mask expects True for padded positions
        #    but note that we used `batch_first=True` in our encoder layer
        encoded = self.transformer_encoder(tokens, 
                                           src_key_padding_mask=mask)
        # encoded: (B, max_tokens, d_model)

        # 4) Unproject, unpad, unflatten
        graph = self._unflatten(encoded, metadata)  # returns updated graph: nodes/edges/globals

        # 5) Optionally decode with MLP decoders
        # Note: the unflatten step has assigned graph['nodes'], graph['edges'], graph['globals']
        # to the new embeddings. Let's decode them if the user provided decoders.
        if hasattr(self, 'node_decoder'):
            graph['nodes'] = self.node_decoder(graph['nodes'])
        if hasattr(self, 'edge_decoder'):
            graph['edges'] = self.edge_decoder(graph['edges'])
        if hasattr(self, 'global_decoder'):
            # shape: [#graphs, global_dim]
            graph['globals'] = self.global_decoder(graph['globals'])
        
        graph['n_node'] = in_graph['n_node']
        graph['n_edge'] = in_graph['n_edge']
        graph['receivers'] = in_graph["receivers"]
        graph['senders'] = in_graph["senders"]

        return [graph]

    ###############
    # HELPER FUNCTIONS
    ###############
    def _flatten_and_pad(self, graph: Dict):
        """
        1) Suppose we have a batch of graphs. Each graph has 
           - a variable number of nodes
           - a variable number of edges
           - exactly 1 global feature (?), or None
        2) We flatten node embeddings, edge embeddings, and optional global into a single sequence.
        3) We do this for each graph in the batch, then pad to the maximum token length.

        Returns:
            tokens: Tensor of shape (B, max_tokens, feature_dim)  [before projection]
            mask:   Bool Tensor of shape (B, max_tokens) indicating padding
            metadata: extra info to help unflatten
        """
        # This is a schematic; adapt to your data structure.
        # Let's assume graph['n_node'] and graph['n_edge'] 
        # are arrays telling how many nodes/edges each graph has.

        n_node_per_graph = graph['n_node']  # shape [B]
        n_edge_per_graph = graph['n_edge']  # shape [B]

        # Convert to CPU or keep on device
        n_graphs = n_node_per_graph.shape[0]

        nodes = graph['nodes']   # shape [sum(n_node), node_dim]
        edges = graph['edges']   # shape [sum(n_edge), edge_dim]
        globals_ = graph['globals']  # shape [B, global_dim] or None

        # We'll build tokens per graph
        # e.g., for graph i:
        #   # of node tokens = n_node_per_graph[i]
        #   # of edge tokens = n_edge_per_graph[i]
        #   # + 1 global token (optional)

        # Indices for slicing
        node_idx = 0
        edge_idx = 0

        # We find the maximum sequence length among the B graphs
        max_len = 0
        seqs = []
        for i in range(n_graphs):
            num_nodes_i = n_node_per_graph[i].item()
            num_edges_i = n_edge_per_graph[i].item()

            seq_len = num_nodes_i + num_edges_i
            if globals_ is not None:
                seq_len += 1
            max_len = max(max_len, seq_len)

        # Prepare placeholders
        # We'll store float('nan') for padding or just zeros. We'll use a mask to indicate padding
        feature_dim = nodes.shape[1]  # before projection
        tokens = torch.zeros((n_graphs, max_len, feature_dim), 
                             device=nodes.device, dtype=nodes.dtype)
        # True means "this is a PAD position"
        mask = torch.ones((n_graphs, max_len), device=nodes.device, dtype=torch.bool)

        # We'll keep metadata to reconstruct
        metadata = {
            'max_len': max_len,
            'feature_dim': feature_dim,
            'has_globals': (globals_ is not None),
            'node_idx_slices': [],
            'edge_idx_slices': [],
        }

        for i in range(n_graphs):
            num_nodes_i = n_node_per_graph[i].item()
            num_edges_i = n_edge_per_graph[i].item()

            # slice nodes
            node_slice = nodes[node_idx:node_idx + num_nodes_i, :]
            node_idx += num_nodes_i

            # slice edges
            edge_slice = edges[edge_idx:edge_idx + num_edges_i, :]
            edge_idx += num_edges_i

            # Combine them
            combined = torch.cat([node_slice, edge_slice], dim=0)  # shape [n_i + e_i, feat_dim]
            # Optionally add global
            if globals_ is not None:
                global_token = globals_[i:i+1, :]  # shape [1, feat_dim]
                combined = torch.cat([combined, global_token], dim=0)

            seq_len_i = combined.shape[0]
            tokens[i, :seq_len_i, :] = combined
            mask[i, :seq_len_i] = False  # not padded

            # We'll store how to unflatten
            metadata['node_idx_slices'].append((0, num_nodes_i))
            metadata['edge_idx_slices'].append((num_nodes_i, num_nodes_i + num_edges_i))
            # We'll infer global as the last token if has_globals.

        return tokens, mask, metadata

    def _unflatten(self, encoded: torch.Tensor, metadata: Dict) -> Dict:
        """
        Now, split back into node embeddings, edge embeddings, global embeddings (if any).
        encoded shape: (B, max_len, d_model)
        We'll build a new graph dict with updated features.
        """
        B, max_len, _ = encoded.shape
        device = encoded.device

        # We'll create placeholders for final node/edge/global
        node_embs = []
        edge_embs = []
        global_embs = [] if metadata['has_globals'] else None

        for i in range(B):
            (node_start, node_end) = metadata['node_idx_slices'][i]
            (edge_start, edge_end) = metadata['edge_idx_slices'][i]

            # slice: encoded[i, :some_len, :]
            # figure out how many total tokens for that graph
            # we can check mask or use the slices to see how many we actually used
            num_tokens_i = edge_end
            if metadata['has_globals']:
                num_tokens_i += 1

            sub = encoded[i, :num_tokens_i, :]  # shape [num_tokens_i, d_model]
            # node slice
            node_slice = sub[node_start:node_end, :]  # shape [num_nodes_i, d_model]
            edge_slice = sub[edge_start:edge_end, :]  # shape [num_edges_i, d_model]
            global_slice = None
            if metadata['has_globals']:
                global_slice = sub[-1:, :]  # last token

            node_embs.append(node_slice)
            edge_embs.append(edge_slice)
            if metadata['has_globals']:
                global_embs.append(global_slice)

        # Now we need to concat them into a single big tensor for nodes / edges
        graph = {}
        graph['nodes'] = torch.cat(node_embs, dim=0) if len(node_embs) > 0 else None
        graph['edges'] = torch.cat(edge_embs, dim=0) if len(edge_embs) > 0 else None
        if metadata['has_globals']:
            graph['globals'] = torch.cat(global_embs, dim=0)  # shape [B, d_model]
        else:
            graph['globals'] = None

        # We must preserve graph['n_node'], graph['n_edge'], etc. from the original input
        # so that subsequent steps know how to re-split. 
        # Let's assume we store them in the final graph as well. 
        # (In your real code, you might pass in the original graph dict and update it.)
        return graph
    
class TransformerGraphModelPC(TransformerGraphModel):
    """Encode-process-decode GNN architecture."""

    def __init__(self, dims: List[int], pointnet_ls_list: List[int],
                    vec_ind: List[int], pcd_ind: List[int], norm_ind: List[int],
                    d_model: int, nhead: int, num_tf_layers: int,
                 **kwargs: List[int]) -> None:
        super().__init__(dims, d_model, nhead, num_tf_layers, **kwargs)
        self.merge = MLP(kwargs['node_encoder_layers'], kwargs['node_encoder_layers'][-1] \
                            + pointnet_ls_list[-1])
        # note this is different from vanilla GNN
        self.params.append(self.merge.parameters())
        self.vec_ind = vec_ind
        self.pcd_ind = pcd_ind
        self.norm_ind = norm_ind
        self.pcd_feat = PointNetEncoder(pointnet_ls_list, \
                                    global_feat=True, \
                                    feature_transform=True, \
                                    channel=6)
        self.params.append(self.pcd_feat.parameters())

    def forward(self, graph: Dict) -> List[Dict]:
        """Torch forward model."""
        in_graph = graph.copy()
        if hasattr(self, 'node_encoder'):
            pcd_input = graph['nodes'][:, self.pcd_ind]
            vec_input = graph['nodes'][:, self.vec_ind]
            norm_input = graph['nodes'][:, self.norm_ind]
            B = graph['nodes'].shape[0]
            pcd_input = pcd_input.view(B, -1, 3)
            norm_input = norm_input.view(B, -1, 3)
            pc_input = torch.cat([pcd_input, norm_input], dim=2)
            pc_input = pc_input.permute(0, 2, 1)
            pc_feat = self.pcd_feat(pc_input)
            vec_feat = self.node_encoder(vec_input)
            graph['nodes'] = self.merge(torch.cat([vec_feat, pc_feat], dim=1))
        if hasattr(self, 'edge_encoder'):
            graph['edges'] = self.edge_encoder(graph['edges'])
        if hasattr(self, 'global_encoder'):
            if graph['globals'] is not None:
                if graph['globals'].shape[1] == 0:
                    # handel empty globals
                    graph['globals'] = torch.ones_like(graph['n_node'])\
                        .to(graph['nodes'].dtype)
                graph['globals'] = self.global_encoder(graph['globals'])
            else:
                graph['globals'] = None

        # 2) Flatten nodes, edges, globals into tokens
        tokens, mask, metadata = self._flatten_and_pad(graph)  
        # 3) Pass through Transformer as message passing
        #    src_key_padding_mask expects True for padded positions
        #    but note that we used `batch_first=True` in our encoder layer
        encoded = self.transformer_encoder(tokens, 
                                           src_key_padding_mask=mask)
        # encoded: (B, max_tokens, d_model)

        # 4) Unproject, unpad, unflatten
        graph = self._unflatten(encoded, metadata)  # returns updated graph: nodes/edges/globals

        # 5) Optionally decode with MLP decoders
        # Note: the unflatten step has assigned graph['nodes'], graph['edges'], graph['globals']
        # to the new embeddings. Let's decode them if the user provided decoders.
        if hasattr(self, 'node_decoder'):
            graph['nodes'] = self.node_decoder(graph['nodes'])
        if hasattr(self, 'edge_decoder'):
            graph['edges'] = self.edge_decoder(graph['edges'])
        if hasattr(self, 'global_decoder'):
            # shape: [#graphs, global_dim]
            graph['globals'] = self.global_decoder(graph['globals'])

        graph['n_node'] = in_graph['n_node']
        graph['n_edge'] = in_graph['n_edge']
        graph['receivers'] = in_graph["receivers"]
        graph['senders'] = in_graph["senders"]

        return [graph]

def setup_tf_net(graph_dataset: GraphDictDataset, tf_dim: int,
                    gnn_layer_size: int, num_layers: int):
    dims = _setup_dims(graph_dataset)
    include_globals = dims[-1] > 0

    enc_dims = [gnn_layer_size, gnn_layer_size]
    enc_layers = {
        'nodes': [gnn_layer_size, enc_dims[0]],
        'edges': [gnn_layer_size, enc_dims[1]]
    }
    if include_globals:
        enc_layers['globals'] = [gnn_layer_size, enc_dims[1]]


    dec_layers = {'nodes': [dims[3]], 'edges': [dims[4]]}
    if include_globals:
        dec_layers['globals'] = [dims[-1]]
    layer_dict = {}

    layer_dict['node_encoder_layers'] = enc_layers['nodes']
    layer_dict['edge_encoder_layers'] = enc_layers['edges']
    if include_globals:
        layer_dict['global_encoder_layers'] = enc_layers['globals']

    layer_dict['node_decoder_layers'] = dec_layers['nodes']
    layer_dict['edge_decoder_layers'] = dec_layers['edges']
    if include_globals:
        layer_dict['global_decoder_layers'] = dec_layers['globals']

    model = TransformerGraphModel(dims, 
            d_model=tf_dim, 
            nhead=4, 
            num_tf_layers=num_layers,
            **layer_dict)
    return model

def setup_pointnet_tf_net(graph_dataset: GraphDictDataset, 
                          pointnet_feat_lst: List[int],
                        node_feature_to_index: Dict,
                        tf_dim: int,
                        gnn_layer_size: int, num_layers: int):
    dims, vec_ind, pcd_ind, norm_ind = _setup_dims_points(graph_dataset, node_feature_to_index)
    include_globals = dims[-1] > 0

    enc_dims = [gnn_layer_size, gnn_layer_size]
    enc_layers = {
        'nodes': [gnn_layer_size, enc_dims[0]],
        'edges': [gnn_layer_size, enc_dims[1]]
    }
    if include_globals:
        enc_layers['globals'] = [gnn_layer_size, enc_dims[1]]


    dec_layers = {'nodes': [dims[3]], 'edges': [dims[4]]}
    if include_globals:
        dec_layers['globals'] = [dims[-1]]
    layer_dict = {}

    layer_dict['node_encoder_layers'] = enc_layers['nodes']
    layer_dict['edge_encoder_layers'] = enc_layers['edges']
    if include_globals:
        layer_dict['global_encoder_layers'] = enc_layers['globals']

    layer_dict['node_decoder_layers'] = dec_layers['nodes']
    layer_dict['edge_decoder_layers'] = dec_layers['edges']
    if include_globals:
        layer_dict['global_decoder_layers'] = dec_layers['globals']

    model = TransformerGraphModelPC(dims, 
            pointnet_feat_lst, vec_ind, pcd_ind, norm_ind,
            d_model=tf_dim, 
            nhead=4, 
            num_tf_layers=num_layers,
            **layer_dict)
    return model