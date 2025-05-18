from data import MultiMolGraphDataset
from torch_geometric.loader import DataLoader
import torch
import torchmetrics
from torchmetrics import MeanAbsoluteError

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
sdf_path = '/home/calvin/code/chemprop_phd_customised/habnet/data/processed/sdf_data'
target_data = '/home/calvin/code/chemprop_phd_customised/habnet/data/processed/target_data/target_data.csv'

# Read in target_data with pandas and then select 'psi_1_dihedral' column and convert the dihedral angles to sin and cos 
import pandas as pd
import numpy as np
# 1 & 2. Load your data
target_df = pd.read_csv(target_data)

# Drop -10 rows in the dihedral angle column
target_df = target_df[target_df['psi_1_dihedral'] != -10]

# 3. Extract the dihedral angles
angles = target_df['psi_1_dihedral']

# 3.1 Remove any -10 rows in the dihedral angle column
angles = angles[angles != -10]
# 3.2 Remove any NaN values
angles = angles.dropna()

# 4. If angles are in degrees, convert to radians
angles_rad = np.deg2rad(angles)

# 5. Compute sin & cos and assign
target_df['psi_1_dihedral_sin'] = np.sin(angles_rad)
target_df['psi_1_dihedral_cos'] = np.cos(angles_rad)

# Optional: inspect
print(target_df[['psi_1_dihedral', 'psi_1_dihedral_sin', 'psi_1_dihedral_cos']].head())



# 6. Save the modified DataFrame back to CSV
target_data = '/home/calvin/code/chemprop_phd_customised/habnet/data/processed/target_data/target_data_sin_cos.csv'
target_df.to_csv(target_data, index=False)

mol_dataset = MultiMolGraphDataset(
    root='.',
    sdf_folder=sdf_path,
    input_type=['r1h', 'r2h'],
    target_csv=target_data,
    target_columns=['psi_1_dihedral_sin', 'psi_1_dihedral_cos'],
    keep_hs=True,
    sanitize=False,
    force_reload=True
)

# Split the dataset into train and test sets and val sets [0.8, 0.1, 0.1]
train_size = int(0.8 * len(mol_dataset))
test_size = int(0.1 * len(mol_dataset))
val_size = len(mol_dataset) - train_size - test_size

import torch
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
    mol_dataset,
    [train_size, test_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 4,
    shuffle    = True,
    follow_batch = ['x_s', 'x_t']    # tells PyG to create x_s_batch, x_t_batch
)
test_loader = DataLoader(
    test_dataset,
    batch_size = 32,
    shuffle    = False,
    follow_batch = ['x_s', 'x_t']    # tells PyG to create x_s_batch, x_t_batch
)
val_loader = DataLoader(
    val_dataset,
    batch_size = 32,
    shuffle    = False,
    follow_batch = ['x_s', 'x_t']    # tells PyG to create x_s_batch, x_t_batch
)

from torch_geometric.data import Batch, Data

from torch_geometric.data import Batch, Data

from torch_geometric.data import Data, Batch

def extract_source_batch(pair_batch):
    data_list = []

    for i in range(pair_batch.ptr.numel() - 1):
        start = pair_batch.ptr[i].item()
        end = pair_batch.ptr[i + 1].item()

        # Mask edges belonging to this graph
        node_mask = torch.arange(pair_batch.x_s.size(0), device=pair_batch.x_s.device)
        node_mask = (node_mask >= start) & (node_mask < end)
        node_indices = node_mask.nonzero(as_tuple=False).view(-1)

        # Map global node indices to local ones
        node_id_map = {old.item(): new for new, old in enumerate(node_indices)}

        edge_src = pair_batch.edge_index_s[0]
        edge_dst = pair_batch.edge_index_s[1]
        edge_mask = node_mask[edge_src] & node_mask[edge_dst]
        edge_indices = edge_mask.nonzero(as_tuple=False).view(-1)

        edge_index = pair_batch.edge_index_s[:, edge_indices]
        edge_index = edge_index.clone()
        edge_attr = pair_batch.edge_attr_s[edge_indices]

        # Relabel edge indices to local ones
        edge_index[0] = edge_index[0].apply_(lambda x: node_id_map[x])
        edge_index[1] = edge_index[1].apply_(lambda x: node_id_map[x])

        data = Data(
            x=pair_batch.x_s[node_indices],
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=torch.zeros(len(node_indices), dtype=torch.long)
        )
        data_list.append(data)

    return Batch.from_data_list(data_list)



print(f"Number of training samples: {len(train_loader.dataset)}")



import torch
import torch.nn.functional as F


def debug_nt_xent_loss(z1, z2, temperature=0.5):
    import torch.nn.functional as F

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # shape: [2N, D]
    sim = torch.mm(z, z.T) / temperature  # shape: [2N, 2N]

    # remove self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)

    # target: each positive is the opposite view
    targets = torch.arange(N, device=z.device)
    targets = torch.cat([targets + N, targets])

    try:
        loss = F.cross_entropy(sim, targets)
    except Exception as e:
        print("Loss computation failed:", e)
        print("Sim shape:", sim.shape)
        print("Targets shape:", targets.shape)
        raise e

    return loss

from gine import GINEEncoder
encoder = GINEEncoder(
    node_in=133,
    edge_in=27,
    edge_proj=True,
    use_batchnorm=False,
    dropout=0.1,

)
encoder = encoder.to(device)

# projector = torch.nn.Sequential(
#     torch.nn.Linear(256, 512),
#     torch.nn.ReLU(),
#     torch.nn.Linear(512, 128)
# )
projector = torch.nn.Linear(256, 128)
projector = projector.to(device)


from torch_geometric.utils import subgraph
from torch_geometric.data import Data
def augment_graphs(data, drop_rate=0.05):
    num_nodes = data.num_nodes

    # Randomly drop nodes
    keep_mask = torch.rand(num_nodes, device=data.x.device) > drop_rate
    keep_indices = keep_mask.nonzero(as_tuple=False).view(-1)

    if keep_indices.numel() == 0:
        # Avoid creating an empty graph
        keep_indices = torch.tensor([0], device=data.x.device)

    edge_index, edge_attr = subgraph(
        keep_indices,
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True,
        num_nodes=num_nodes
    )

    # Create new graph
    new_data = Data(
        x=data.x[keep_indices],
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=data.batch[keep_indices]
    )
    return new_data


batch = next(iter(train_loader))
print(batch.num_graphs)  # Should be > 1

batch = extract_source_batch(batch).to(device)
print(batch)

view1 = augment_graphs(batch, drop_rate=0.05)
view2 = augment_graphs(batch , drop_rate=0.05)


h1 = encoder(view1)
print("encoder(view1) shape:", h1.shape)

z1 = projector(h1)
print("projector(encoder(view1)) shape:", z1.shape)


h2 = encoder(view2)
print("encoder(view2) shape:", h2.shape)
z2 = projector(h2)
print("projector(encoder(view2)) shape:", z2.shape)
print("z1 shape:", z1.shape)
print("z2 shape:", z2.shape)
