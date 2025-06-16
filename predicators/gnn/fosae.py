from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from predicators.gnn.gnn_utils import _create_super_graph, _convert_to_data, split_graphs

############################
# Gumbel-Softmax (re-used)
############################
def gumbel_softmax(logits: torch.Tensor, tau: float=1.0, hard: bool=True) -> torch.Tensor:
    """
    Gumbel-Softmax trick to get a discrete-like one-hot sample from 'logits'.
      - tau is temperature.
      - if hard=True, we take an argmax in the forward pass 
        but allow gradients via the 'soft' sample.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = logits + gumbel_noise
    # Softmax across N dimension
    y = F.softmax(y / tau, dim=-1)

    if hard:
        # Take argmax (one-hot) in the forward pass
        shape = y.size()
        _, max_idx = y.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, max_idx, 1.0)
        # Straight-through gradient
        y = (y_hard - y).detach() + y
    
    return y

############################
# DotProductAttention
############################
class DotProductAttention(nn.Module):
    """
    A single-head dot-product attention that:
      1) Learns transformations (Wq, Wk, Wv) for the input features of dimension F.
      2) Uses a 'global query' derived from x (e.g., the mean over objects).
      3) Computes attention distribution over N objects, optionally Gumbel-softmax for discrete selection.

    x: shape (batch_size, N, F)
    returns: 
      att_dist: shape (batch_size, N) (the attention weights, ~ a distribution),
      context: shape (batch_size, F)   (the attended combination of objects).
    """
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        # Linear maps for Q, K, V
        self.Wq = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(feature_dim, hidden_dim, bias=False)
        
        # Final projection back to feature_dim if you want context in the same dimension
        self.out_proj = nn.Linear(hidden_dim, feature_dim, bias=False)
        
    def forward(self, x: torch.Tensor, tau: float=1.0, hard: bool=False) -> torch.Tensor:
        """
        x: (batch_size, N, F)
        tau, hard: for Gumbel-Softmax. If you want purely continuous attention, you can skip Gumbel.

        returns:
          att_dist (batch_size, N) : attention distribution over N objects
          context (batch_size, F)  : the weighted sum of values (in original F-dim)
        """
        bs, N, F_ = x.shape
        
        # 1) Derive a "global query" from the input.
        #    For example, just take the mean of x across N, shape = (bs, F).
        #    Then transform it by Wq to get Q shape = (bs, hidden_dim).
        x_mean = x.mean(dim=1)               # (bs, F)
        Q = self.Wq(x_mean)                  # (bs, hidden_dim)

        # 2) Key and Value for each object
        #    K, V shape = (bs, N, hidden_dim)
        K = self.Wk(x)
        # V is not used in this function, but you can use it for context if you want.
        V = self.Wv(x)

        # 3) Compute dot-product QK^T (scaled by sqrt(hidden_dim))
        #    Q : (bs, hidden_dim) => unsqueeze(1) => (bs, 1, hidden_dim)
        #    K : (bs,  N, hidden_dim)
        #    => att_logits shape = (bs, 1, N)
        att_logits = torch.bmm(Q.unsqueeze(1), K.transpose(1,2))  # (bs, 1, N)
        att_logits = att_logits / math.sqrt(K.size(-1))           # scale

        # 4) Squeeze to (bs, N)
        att_logits = att_logits.squeeze(1)    # (bs, N)

        # 5) Optionally Gumbel-softmax for discrete 1-hot or softmax for continuous
        #    If you want a discrete selection, set hard=True.
        #    If you want a continuous distribution, keep hard=False (and possibly skip Gumbel).
        att_dist = gumbel_softmax(att_logits, tau=tau, hard=hard) # (bs, N)

        return att_dist

############################
# PredicateNetwork
############################
class PredicateNetwork(nn.Module):
    """
    A 2-layer MLP that takes the A arguments (concatenated),
    then outputs a 1-hot boolean {0,1} using Gumbel-Softmax.
    """
    def __init__(self, arg_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(arg_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 2 = 'True' or 'False'
        
    def forward(self, args: torch.Tensor, tau: float=1.0, hard: bool=True) -> torch.Tensor:
        """
        args: shape (batch_size, A*F). The concatenated arguments for one PU.
        returns: shape (batch_size, 2) -> Gumbel-Softmax -> one-hot of dimension=2
        """
        h = F.relu(self.fc1(args))
        logits = self.fc2(h)
        return gumbel_softmax(logits, tau=tau, hard=hard)

############################
# Full Relational AutoEncoder
############################
class RelationalAutoEncoder(nn.Module):
    """
    The high-level model with:
      - U PUs
      - A attentions per PU
      - P predicates (shared across PUs)
      - Decoder reconstructing NxF
    """
    def __init__(
        self,
        max_n: int,    # Maximum number of objects in the graph, this is required for decoder
        F: int,         # Feature dimension of each object
        U: int,         # Number of PUs
        A: int,         # Number of attentions per PU
        P: int,         # Number of predicate types (PNs)
        hidden_att_dim: int = 64,
        hidden_pred_dim: int = 64,
        hidden_dec_dim: int = 128
    ):
        super().__init__()
        
        self.F = F
        self.U = U
        self.A = A
        self.P = P
        self.max_n = max_n

        # Create A distinct attention networks per PU:
        # but often we share or let them be distinct across PUs.
        # For demonstration, let's keep them distinct for each (u,a).
        # So we'll have U*A attention nets total.
        self.attention_nets = nn.ModuleList()
        for u in range(U):
            for a in range(A):
                net = DotProductAttention(feature_dim=F, hidden_dim=hidden_att_dim)
                self.attention_nets.append(net)
        
        # Create P predicate networks (shared across PUs!)
        # Each PN input dimension = A*F (concatenate A object embeddings)
        self.predicates = nn.ModuleList()
        for p in range(P):
            pred_net = PredicateNetwork(arg_dim=A*F, hidden_dim=hidden_pred_dim)
            self.predicates.append(pred_net)
        
        # A simple decoder: from (U x P) boolean outputs -> NxF reconstruction
        # Flatten the (U,P) booleans => dimension = U*P*2 if we keep 1-hot,
        # or U*P if we interpret them as a single bit. We'll assume 2-dim 1-hot for clarity.
        # So the input to the decoder is dimension = U*P*2.
        decoder_input_dim = U * P * 2
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dec_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dec_dim),
            nn.Linear(hidden_dec_dim, hidden_dec_dim),
            nn.ReLU(),
            nn.Linear(hidden_dec_dim, max_n * F),
            nn.Sigmoid()  # ensures output is in [0,1]
        )
        
    def forward(self, graph: dict, 
                tau: float=1.0, hard: bool=True) -> torch.Tensor:
        """
        x: (B*N, F), note that we flatten the batch and object dimensions as a graph.
        1) For each PU (u) and attention index (a), get att_{u,a}(x) => shape (bs, N)
        2) Dot-product with x to get each argument g_{u,a}
        3) For each PU (u), feed the concatenated arguments to each of the P predicate nets (shared),
           get P booleans => shape (bs, P, 2)
        4) Collect all booleans from all PUs => shape (bs, U, P, 2)
        5) Flatten => shape (bs, U*P*2) => decode => shape (bs, N*F)
        6) Reshape => (bs, N, F)
        """
        # Step (0): Extract the input nodes from the graph, and reshape to desired shape
        original_nodes = graph['nodes']
        bs = len(graph['n_node'])
        input_x = torch.zeros(bs, self.max_n, self.F).to(original_nodes.device)
        # reshape the input nodes to desired shape, zero-padding if necessary
        # input_x: (bs, N, F)
        start_node = 0
        for i in range(bs):
            end_node = start_node + graph['n_node'][i]
            input_x[i, :end_node-start_node] = original_nodes[start_node:end_node]
            start_node = end_node
        
        # Step (5) Flatten => shape (bs, U*P*2)
        z = self.encode(input_x, bs, tau=tau, hard=hard)
        
        # Step (5 continued) pass to decoder => reconstruct => (bs, N*F)
        reconstruction = self.decoder(z)
        
        # Step (6) reshape => (bs, N, F)
        reconstruction = reconstruction.view(bs, self.max_n, self.F)
        reconstruct_nodes = torch.zeros_like(original_nodes)
        start_node = 0
        for b in range(bs):
            end_node = start_node + graph['n_node'][b]
            # copy the reconstructed nodes to the original nodes
            reconstruct_nodes[start_node:end_node] = reconstruction[b, :end_node-start_node]
            start_node = end_node
        return reconstruct_nodes
    
    def encode(self, input_x: torch.Tensor, bs: int,
               tau: float=1.0, hard: bool=True) -> torch.Tensor:
        """
        Encode the input graph into a latent representation.
        """
        # Step (1) & (2): compute arguments via attention
        # We'll store arguments in g[u,a] for each PU and argument index
        # We'll keep them in a list for clarity
        arguments = []
        
        # We have U*A attention nets in self.attention_nets, in row-major for (u,a).
        # i.e. net_index = u*A + a
        for u in range(self.U):
            g_u = []
            for a in range(self.A):
                net_index = u * self.A + a
                att_dist = self.attention_nets[net_index](input_x, tau=tau, hard=hard)  # (bs, N)
                
                # Dot-product with x: shape = (bs, N) x (bs, N, F)
                # We'll do a batch-wise multiply + sum. 
                # att_dist.unsqueeze(-1) => (bs, N, 1)
                # x => (bs, N, F)
                # elementwise => (bs, N, F), sum over dim=1 => (bs, F)
                g_ua = (att_dist.unsqueeze(-1) * input_x).sum(dim=1)  # (bs, F)
                
                g_u.append(g_ua)
            
            # Concat the A arguments => shape (bs, A*F)
            g_u_cat = torch.cat(g_u, dim=-1)
            arguments.append(g_u_cat)
        
        # Step (3) & (4): for each PU, feed to each predicate -> collect booleans
        # We'll produce shape (bs, U, P, 2)
        all_booleans = []
        for u in range(self.U):
            # arguments[u] => shape (bs, A*F)
            g_u_cat = arguments[u]
            
            # For each of the P predicate nets (shared across PUs)
            # we get shape (bs, 2) from each, stack them -> shape (bs, P, 2)
            bools_pu = []
            for p_i in range(self.P):
                bool_p = self.predicates[p_i](g_u_cat, tau=tau, hard=hard)  # (bs, 2)
                bools_pu.append(bool_p.unsqueeze(1))  # (bs, 1, 2)
            
            bools_pu = torch.cat(bools_pu, dim=1)  # (bs, P, 2)
            all_booleans.append(bools_pu.unsqueeze(1))  # (bs, 1, P, 2)
        
        # Combine along PU dimension => (bs, U, P, 2)
        all_booleans = torch.cat(all_booleans, dim=1)
        
        # Step (5) Flatten => shape (bs, U*P*2)
        z = all_booleans.reshape(bs, self.U * self.P * 2)

        return z

def setup_sae_net(
        max_n: int,
        num_features: int,
        model_setting_dict: dict
    ) -> RelationalAutoEncoder:
    """
    Setup the SAE model with the given settings.
    """
    # Extract settings
    num_pus = model_setting_dict['num_pus']
    num_attentions = model_setting_dict['num_attentions']
    num_predicates = model_setting_dict['num_predicates']
    hidden_att_dim = model_setting_dict['hidden_att_dim']
    hidden_pred_dim = model_setting_dict['hidden_pred_dim']
    hidden_dec_dim = model_setting_dict['hidden_dec_dim']
    model = RelationalAutoEncoder(
        max_n=max_n,
        F=num_features,
        U=num_pus,
        A=num_attentions,
        P=num_predicates,
        hidden_att_dim=hidden_att_dim,
        hidden_pred_dim=hidden_pred_dim,
        hidden_dec_dim=hidden_dec_dim
    )
    return model

def get_single_model_prediction(ama_model: Any,
                                sae_model: Any,
                                in_graph_ama: Dict,
                                in_graph_sae: Dict,
                                device: Optional[torch.device] = None) -> Dict:
    """Get a prediction from the given model on the given input."""
    if device is not None:
        ama_model.to(device)
        sae_model.to(device)
    sae_model.eval()
    ama_model.eval()
    num_objects = in_graph_sae['n_node']
    with torch.no_grad():
        input_x = torch.zeros(1, sae_model.max_n, \
            sae_model.F).to(device)
        input_x[0, :num_objects, :] = \
            torch.tensor(in_graph_sae["nodes"]).to(device)
        z = sae_model.encode(input_x, 1)
        z_dim = z.shape[-1]
        
        inputs = _create_super_graph([in_graph_ama], device=device)
        if device is not None:
            inputs = {key: val.to(device) for key, val in inputs.items()}
        inputs["globals"][:, -z_dim:] = z[0]
        outputs = ama_model(inputs.copy())
        graphs = split_graphs(_convert_to_data(outputs[-1]))
        assert len(graphs) == 1
        graph = graphs[0]
        graph['nodes'] = graph['nodes'].numpy()
        graph['senders'] = graph['senders'].numpy()
        graph['receivers'] = graph['receivers'].numpy()
        graph['edges'] = graph['edges'].numpy()
        if graph['globals'] is not None:
            graph['globals'] = graph['globals'].numpy()
        graph['n_node'] = graph['n_node'].item()
        graph['n_edge'] = graph['n_edge'].item()
    return graph

############################
# Usage Example
############################
if __name__ == "__main__":
    # Suppose we have N=5 objects, F=4 dims, U=2 PUs, A=2 attentions, P=3 predicates
    N, F = 5, 4
    U, A, P = 2, 2, 3
    batch_size = 8
    
    # Create random input
    x = torch.randn(batch_size, N, F)
    
    # Build model
    model = RelationalAutoEncoder(N=N, F=F, U=U, A=A, P=P)
    
    # Forward pass
    recon = model(x, tau=1.0, hard=True)
    print("recon shape:", recon.shape)  # Expect (8, 5, 4)
    
    # The loss would be an MSE between recon and x:
    loss = F.mse_loss(recon, x)
    loss.backward()
    
    print("loss:", loss.item())
