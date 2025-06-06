import torch
from torch_geometric.data import Data
import torch_geometric
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from dimenet import DimeNetPPEncoder
def make_view(x, ei, ea, b):
    return Data(x=x, edge_index=ei, edge_attr=ea, batch=b)

class SiamesePairwise(torch.nn.Module):
    def __init__(self, encoder, n_targets=2, fusion='cat',
                 dropout=0.1, head_hidden_dims=[256,128]):
        super().__init__()
        self.encoder = encoder
        D = encoder.out_dim

        # compute input dim to head
        if fusion == 'cat':
            in_head = 2*D
        elif fusion == 'diff':
            in_head = 3*D
        elif fusion == 'diff2':
            in_head = 4*D
        elif fusion == 'symm':
            in_head = 4*D
        dims   = [in_head, *head_hidden_dims, n_targets]

        layers = []
        for i in range(len(dims)-1):
            layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers += [torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        self.head   = torch.nn.Sequential(*layers)
        self.fusion = fusion

    def _fuse(self, h1, h2):
        if self.fusion == 'cat':
            return torch.cat([h1, h2], dim=-1)
        elif self.fusion == 'symm':
            return torch.cat([h1, h2, (h1-h2).abs(), h1*h2], dim=-1)
        elif self.fusion == 'diff':
            return torch.cat([h1-h2, (h1-h2).abs(), h1*h2], dim=-1)
        elif self.fusion == 'diff2':
            return torch.cat([h1-h2, (h1-h2).abs(), h1*h2, h1+h2], dim=-1)

    def forward(self, pair):
        # build the two graph views
        A = Data(x=pair.x_s, edge_index=pair.edge_index_s,
                 edge_attr=pair.edge_attr_s, batch=pair.x_s_batch)
        B = Data(x=pair.x_t, edge_index=pair.edge_index_t,
                 edge_attr=pair.edge_attr_t, batch=pair.x_t_batch)

        # **just** get graph vectors
        hA = self.encoder(A)   # [batch, D]
        hB = self.encoder(B)   # [batch, D]

        z  = self._fuse(hA, hB)
        out= self.head(z)
        # normalize sin/cos
        return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-7)



class SiamesePairwiseAttn(torch.nn.Module):
    def __init__(self, encoder, n_targets=2, fusion='cat',
                 dropout=0.1, head_hidden_dims=[256,128], n_heads=4):
        super().__init__()
        self.encoder   = encoder
        D = encoder.out_dim
        self.cross_attn =torch.nn.MultiheadAttention(D, n_heads,
                                                dropout=dropout,
                                                batch_first=True)
        # gates start at 0 → inject 50% of the delta at init
        self.alpha_h1 =torch.nn.Parameter(torch.zeros(()))
        self.alpha_h2 =torch.nn.Parameter(torch.zeros(()))

        in_head = 2*D if fusion=='cat' else 4*D
        dims    = [in_head, *head_hidden_dims, n_targets]
        layers  = []
        for i in range(len(dims)-1):
            layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers += [torch.nn.ReLU(),torch.nn.Dropout(dropout)]
        self.head =torch.nn.Sequential(*layers)
        self.fusion = fusion

    def _fuse(self, h1, h2):
        if self.fusion=='cat':
            return torch.cat([h1, h2], dim=-1)
        else:
            return torch.cat([h1, h2, (h1-h2).abs(), h1*h2], dim=-1)

    def forward(self, pair):
        # 1) unpack & run encoder *without* pooling
        A = make_view(pair.x_s, pair.edge_index_s,
                      pair.edge_attr_s, pair.x_s_batch)
        B = make_view(pair.x_t, pair.edge_index_t,
                      pair.edge_attr_t, pair.x_t_batch)

        # assume your encoder signature is now:
        #   node_feats, batch = encoder(data, return_nodes=True)
        hA_nodes, batchA = self.encoder(A, return_nodes=True)
        hB_nodes, batchB = self.encoder(B, return_nodes=True)

        # 2) pack into dense tensors + masks
        hA_seq, maskA = to_dense_batch(hA_nodes, batchA)  # (B, N1, D), bool mask
        hB_seq, maskB = to_dense_batch(hB_nodes, batchB)  # (B, N2, D), bool mask

        # 3) cross-attention: every node in A reads all nodes in B (and vice versa)
        #    key_padding_mask expects shape (B, N2), True == masked out
        attA, _ = self.cross_attn(
            query=hA_seq,
            key=hB_seq,
            value=hB_seq,
            key_padding_mask=~maskB
        )
        attB, _ = self.cross_attn(
            query=hB_seq,
            key=hA_seq,
            value=hA_seq,
            key_padding_mask=~maskA
        )

        # 4) mean-pool *with* mask back to graph vectors
        #    sum only over valid positions, divide by mask.sum
        sumA = (attA * maskA.unsqueeze(-1)).sum(dim=1)
        cntA = maskA.sum(dim=1, keepdim=True)
        hA_new = sumA / cntA.clamp(min=1)

        sumB = (attB * maskB.unsqueeze(-1)).sum(dim=1)
        cntB = maskB.sum(dim=1, keepdim=True)
        hB_new = sumB / cntB.clamp(min=1)

        # 5) original graph vectors (same pooling of un-attended nodes)
        sumA0 = (hA_seq * maskA.unsqueeze(-1)).sum(dim=1)
        hA0   = sumA0 / cntA.clamp(min=1)
        sumB0 = (hB_seq * maskB.unsqueeze(-1)).sum(dim=1)
        hB0   = sumB0 / cntB.clamp(min=1)

        # 6) gated residual
        deltaA = hA_new - hA0
        deltaB = hB_new - hB0
        hA = hA0 + torch.sigmoid(self.alpha_h1) * deltaA
        hB = hB0 + torch.sigmoid(self.alpha_h2) * deltaB

        # 7) fuse & final MLP
        z   = self._fuse(hA, hB)
        out = self.head(z)
        return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-7)

from torch.nn import functional as F
class SiameseDimeNet(torch.nn.Module):
    def __init__(
        self,
        encoder: DimeNetPPEncoder,
        dropout: float = 0.2,
        fusion: str = 'cat',          # 'cat' or 'cat-diff-prod'
        head_hidden_dims=[256, 128],  # new: allow multiple head layers
    ):
        super().__init__()
        self.encoder = encoder
        self.fusion  = fusion

        D = encoder.out_dim
        # determine how big the fused vector is:
        if fusion == 'cat':
            fusion_dim = 2 * D
        else:  # cat-diff-prod
            fusion_dim = 4 * D

        dims = [fusion_dim, *head_hidden_dims, 2]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        self.head = torch.nn.Sequential(*layers)

    def fuse(self, h1, h2):
        if self.fusion == 'cat':
            return torch.cat([h1, h2], dim=-1)
        else:
            return torch.cat([
                h1,
                h2,
                (h1 - h2).abs(),
                h1 * h2,
            ], dim=-1)

    def encode(self, batch):
        batch_s = Data(z=batch.z_s, pos=batch.pos_s, batch=batch.z_s_batch)
        batch_t = Data(z=batch.z_t, pos=batch.pos_t, batch=batch.z_t_batch)
        return self.encoder(batch_s), self.encoder(batch_t)


    def head_and_norm(self, h_fused):
        raw = self.head(h_fused)
        return F.normalize(raw, p=2, dim=-1, eps=1e-8)

    def head_mu_kappa(self, h_fused):
        raw = self.head(h_fused)
        mu    = raw[:, 0]            # unbounded angle prediction (fine)
        kappa = F.softplus(raw[:, 1]) + 1e-3
        kappa = torch.clamp(kappa, max=20)  # or max=50, depending on what works for your loss scale
        return mu, kappa

    def forward(self, batch):
        # This version just calls the pieces in FP32 for now
        h_s, h_t    = self.encode(batch)
        h_fused     = self.fuse(h_s, h_t)
        return self.head_mu_kappa(h_fused)
    



class DimeNet(torch.nn.Module):
    def __init__(
        self,
        encoder,  # DimeNetPPEncoder or compatible GNN
        dropout: float = 0.2,
        head_hidden_dims=[256, 128],  # allow multiple head layers
        uncertainty: bool = False,  # if True, return mu and kappa
    ):
        super().__init__()
        self.encoder = encoder
        self.uncertainty = uncertainty

        # Output dimension of encoder (usually encoder.out_dim)
        D = encoder.out_dim

        # Build head MLP: input dim = D, followed by head_hidden_dims, and output 2 (mu, kappa)
        dims = [D, *head_hidden_dims, 2]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        self.head = torch.nn.Sequential(*layers)

    def encode(self, batch):
        # batch should have z, pos, batch (if batched)
        data = Data(z=batch.z, pos=batch.pos, batch=batch.batch)
        return self.encoder(data)

    def head_and_norm(self, h):
        raw = self.head(h)
        return F.normalize(raw, p=2, dim=-1, eps=1e-8)

    def head_mu_kappa(self, h):
        raw = self.head(h)
        mu    = raw[:, 0]
        kappa = F.softplus(raw[:, 1]) + 1e-3
        kappa = torch.clamp(kappa, max=20)
        return mu, kappa

    def forward(self, batch):
        # Forward pass: encode, then head
        h = self.encode(batch)
        if self.uncertainty:
            return self.head_mu_kappa(h)
        else:
            # If uncertainty is False, just return the normalized output
            return self.head_and_norm(h)




