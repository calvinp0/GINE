from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, Set2Set, GlobalAttention, AttentionalAggregation
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout, LayerNorm
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add


class GINEEncoder(torch.nn.Module):
    def __init__(self,
                 node_in: int   = 115,
                 edge_in: int   = 14,
                 hidden:  int   = 256,
                 depth:   int   = 5,
                 edge_proj:     bool = True,
                 use_batchnorm: bool = True,
                 dropout:        float = 0.1,
                 pooling:        str  = 'mean',
                 processing_steps: int = 3):
        super().__init__()
        # store dropout rate for use in GNN and head
        self.dropout = dropout
        # select pooling strategy
        self.pooling = pooling.lower()

        # -------- optional edge projection ------------
        if edge_proj:
            self.edge_proj = Linear(edge_in, hidden, bias=False)
            edge_dim = hidden
        else:
            self.edge_proj = None
            edge_dim = edge_in

        # -------- optional input projection -----------
        if node_in != hidden:
            self.input_proj = Linear(node_in, hidden)
        else:
            self.input_proj = torch.nn.Identity()

        # -------- GINE layers + (optional) BN ---------
        self.convs, self.bns = torch.nn.ModuleList(), torch.nn.ModuleList()
        mlp = lambda: Sequential(Linear(hidden, hidden), ReLU(),
                                 Linear(hidden, hidden),
                                 LayerNorm(hidden))

        for _ in range(depth):
            self.convs.append(GINEConv(mlp(), edge_dim=edge_dim))
            if use_batchnorm:
                self.bns.append(BatchNorm1d(hidden))
        self.use_batchnorm = use_batchnorm
        self.out_dim = hidden
        # dropout layer for GNN residual outputs, uses user parameter
        self.dropout_gnn = Dropout(self.dropout)

        # -------- pooling and readout based on strategy ---------
        if self.pooling == 'set2set':
            self.pool = Set2Set(hidden, processing_steps=processing_steps)
            pool_dim = 2 * hidden
        elif self.pooling == 'max':
            from torch_geometric.nn import global_max_pool
            self.pool = global_max_pool
            pool_dim = hidden
        elif self.pooling == 'attention_pool':
            gate_nn = Sequential(Linear(hidden, 128), ReLU(), Linear(128, 1))
            nn = Sequential(Linear(hidden, hidden), ReLU())
            self.pool = AttentionalAggregation(gate_nn, nn)
            pool_dim = hidden
        elif self.pooling == 'global_attention':
            gate_nn = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, 1))
            self.pool = GlobalAttention(gate_nn)
            pool_dim = hidden
        else:
            from torch_geometric.nn import global_mean_pool
            self.pool = global_mean_pool
            pool_dim = hidden
        self.readout = Linear(pool_dim, hidden)
        self.out_dim = hidden
        head_layers = [
            Linear(hidden, 512), ReLU(), Dropout(self.dropout),
            Linear(512, 256), ReLU(), Dropout(self.dropout),
            Linear(256, hidden)
        ]
        self.head = Sequential(*head_layers)

        self.embedding_dim = pool_dim

    def forward(self, data, return_nodes: bool = False):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        h = self.input_proj(x.float())
        if self.edge_proj is not None:
            ea = self.edge_proj(ea.float())

        # message passing
        for i, conv in enumerate(self.convs):
            h_new = conv(h, ei, ea)
            if self.use_batchnorm:
                h_new = self.bns[i](h_new)
            h = h + F.relu(h_new)
            h = self.dropout_gnn(h)

        if return_nodes:
            # return raw node embeddings + batch assignment
            return h, data.batch

        # pooling
        g = self.pool(h, data.batch)
        return self.readout(g)


import torch
from torch_scatter import scatter_mean

class EdgeUpdateBlock(torch.nn.Module):
    """
    One message-passing step that updates both edges *and* nodes.

    x : [N, d_node]        node embeddings
    e : [E, d_edge]        edge embeddings  (u -> v, same order as edge_index)
    ei: [2, E]             edge indices
    """
    def __init__(self, d_node: int, d_edge: int, hidden: int = 64, aggr='mean'):
        super().__init__()
        self.msg_src = torch.nn.Linear(d_node, d_edge, bias=False)

        self.phi_e = torch.nn.Sequential(
            torch.nn.Linear(d_edge + 2*d_node, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, d_edge))

        self.phi_v = torch.nn.Sequential(
            torch.nn.Linear(d_node + d_edge, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, d_node))

        self.aggr = aggr

    def forward(self, x, edge_index, e):
        src, dst = edge_index                       # [E]

        # 1) message from source node → edge
        m = self.msg_src(x[src])                    # [E, d_edge]

        # 2) update edge feature
        e_input = torch.cat([x[src], x[dst], e + m], dim=-1)
        e_new   = e + self.phi_e(e_input)           # residual on edges

        # 3) aggregate messages to destination nodes
        if self.aggr == 'mean':
            m_dst = scatter_mean(e_new, dst, dim=0, dim_size=x.size(0))
        else:                                       # 'sum'
            m_dst = scatter_add (e_new, dst, dim=0, dim_size=x.size(0))

        v_input = torch.cat([x, m_dst], dim=-1)
        x_new   = x + self.phi_v(v_input)           # residual on nodes
        return x_new, e_new


class EdgeUpdateEncoder(torch.nn.Module):
    def __init__(self, in_node, in_edge, d_node=128, d_edge=64, L=6):
        super().__init__()
        # initial linear projections
        self.lin_x = torch.nn.Linear(in_node, d_node)
        self.lin_e = torch.nn.Linear(in_edge, d_edge)

        self.layers = torch.nn.ModuleList([
            EdgeUpdateBlock(d_node, d_edge) for _ in range(L)
        ])

        # global pooling → graph embedding
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(d_node, d_node),
            torch.nn.ReLU(),
            torch.nn.Linear(d_node, d_node))

        self.out_dim = d_node                      # so SiamesePairwise knows

    def forward(self, data):
        x, e = self.lin_x(data.x), self.lin_e(data.edge_attr)
        ei   = data.edge_index                     # [2,E]

        for layer in self.layers:
            x, e = layer(x, ei, e)

        # global mean-pool (batch aware)
        if hasattr(data, 'batch'):
            graph_x = scatter_mean(x, data.batch, dim=0)
        else:                                     # single graph
            graph_x = x.mean(dim=0, keepdim=True)
        return self.readout(graph_x)




# ------------------------------------------------------------
class EdgeUpdate(torch.nn.Module):
    """single residual edge-update step"""
    def __init__(self, d_node: int, d_edge: int, hidden: int = 64):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * d_node + d_edge, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, d_edge)
        )

    def forward(self, x, edge_index, e):
        src, dst = edge_index
        inp = torch.cat([x[src], x[dst], e], dim=-1)   # [E, 2d_node+d_edge]
        return e + self.mlp(inp)                      # residual
# ------------------------------------------------------------
class GINEEncoderEdgeUpd(torch.nn.Module):
    """
    Your old GINEEncoder + one EdgeUpdate in front of every conv layer.
    """
    def __init__(self,
                 node_in: int = 115,
                 edge_in: int = 14,
                 hidden:  int = 256,
                 depth:   int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout

        # optional projections (unchanged)
        self.x_proj = torch.nn.Linear(node_in,  hidden) if node_in  != hidden else torch.nn.Identity()
        self.e_proj = torch.nn.Linear(edge_in, hidden)     if edge_in != hidden else torch.nn.Identity()

        # stacks ----------------------------------------------------------
        self.updates = torch.nn.ModuleList([
            EdgeUpdate(hidden, hidden) for _ in range(depth)
        ])
        mlp = lambda: torch.nn.Sequential(torch.nn.Linear(hidden, hidden),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden, hidden))
        self.convs = torch.nn.ModuleList([
            GINEConv(mlp(), edge_dim=hidden) for _ in range(depth)
        ])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden) for _ in range(depth)])

        self.readout = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout)
        )
        self.out_dim = hidden

    # --------------------------------------------------------------------
    def forward(self, data):
        x, ei, e = data.x.float(), data.edge_index, data.edge_attr.float()

        x = self.x_proj(x)
        e = self.e_proj(e)

        for upd, conv, bn in zip(self.updates, self.convs, self.bns):
            e = upd(x, ei, e) # EdgeUpdate
            x_new = conv(x, ei, e)
            x     = x + F.relu(bn(x_new))
            x     = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, data.batch)
        return self.readout(g)
