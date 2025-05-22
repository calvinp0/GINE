from atom import MultiHotAtomFeaturizer
from bond import MultiHotBondFeaturizer
from rdkit import Chem
import torch
from torch_geometric.data import Data
import os
import glob
import pandas as pd
from torch_geometric.data import InMemoryDataset
import warnings
import hashlib
from typing import Union, Sequence, Any, Callable, Optional
from pathlib import Path
import numpy as np
import ast
import periodictable
from typing import List, Tuple


from torch_geometric.data import Data, Batch

# Precompute basis parameters
RBF_COUNT = 6
SBF_COUNT = 4
CUTOFF = 5.0  # Å
GAMMA_RBF = 10.0
GAMMA_SBF = 10.0

# Generate radial basis centers and angular basis centers
rbf_mu = np.linspace(0, CUTOFF, RBF_COUNT)
sbf_mu = np.linspace(0, np.pi, SBF_COUNT)

# choose offsets that keep everything < 95
OFFSET_ACCEPTOR = 50          # 50–85 safe range
OFFSET_HYDROGEN = 70
OFFSET_DONOR    = 60



STAR2TAG = {
    "*0": 1,      # acceptor heavy (or however you choose)
    "*1": 1,      # acceptor heavy – some files use different subsets
    "*2": 2,      # transferring H
    "*3": 3,      # donor heavy
    "*4": 3,      # donor heavy (adjacent)
    # 0 means “no special role”
}

def extract_tags(df: pd.DataFrame, reaction_id, z) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns two 1-D tensors of length N_atoms:
        z      – atomic numbers (int64)
        tag_id – 0..4  (int64)
    """

    try:
        props_str = df.loc[df['reaction'] == reaction_id, 'mol_properties'].iloc[0]
    except IndexError:
        raise KeyError(f"reaction '{reaction_id}' not found in DataFrame")

    props = ast.literal_eval(props_str)         # dict-of-dicts, keys = atom indices

    tag_tensor = torch.zeros_like(z, dtype=torch.long)

    # 2) iterate over the (idx, meta) pairs directly
    for idx_str, meta in props.items():
        idx  = int(idx_str)                     # '7' → 7
        star = meta.get('label')                # '*2', '*3', …
        tag_tensor[idx] = STAR2TAG.get(star, 0)

    return tag_tensor


class PairData(Data):
    r"""Holds two independent graphs (source → _s, target → _t)."""
    def __inc__(self, key, value, *args, **kwargs):
        # tell PyG how much to shift indices when batching
        if key == 'edge_index_s':
            return self.x_s.size(0)          # #nodes in source graph
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)
    
    def __cat_dim__(self, key, value, *args, **kwargs):
        """
        Specify concatenation dimension for attributes when batching.
        Return None for 'id' so that it is collected as a Python list.
        """
        if key == 'id':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)



class PairTripletData(Data):
    '''
    # Custom Data subclass to handle triplets and batching
    '''
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        if key in ('triplets_i', 'triplets_j', 'triplets_k'):
            # shift by total nodes in source+target
            return self.num_nodes
        if key == 'edge_to_triplet':
            # shift by total edges
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'id':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


atom_f = MultiHotAtomFeaturizer.v1()
bond_f = MultiHotBondFeaturizer()


class MultiMolGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        sdf_folder: str | list[str],
        target_csv: str,
        input_type: list[str],
        target_columns: list[str],
        keep_hs: bool = False,
        sanitize: bool = True,
        transform=None,
        pre_transform=None,
        force_reload: bool = False,
    ):
        # Store config
        self.sdf_folder = (
            sdf_folder if isinstance(sdf_folder, list) else [sdf_folder]
        )
        self.target_csv = target_csv
        self.input_type = input_type
        self.target_columns = target_columns
        self.keep_hs = keep_hs
        self.sanitize = sanitize
        self.force_reload = force_reload

        # 1. Load target DataFrame
        self.target_df = pd.read_csv(self.target_csv)

        # 2. Gather all .sdf paths
        expanded = []
        for p_str in self.sdf_folder:
            p = Path(p_str).resolve()
            if p.is_dir():
                expanded += sorted(p.glob("*.sdf"))
            elif p.suffix.lower() == ".sdf" and p.exists():
                expanded.append(p)
            else:
                raise FileNotFoundError(f"No .sdf files found at {p}")
        self.sdf_paths = expanded

        # 3. Compute cache hash
        h = hashlib.sha1()
        for p in self.sdf_paths:
            h.update(str(p).encode())
            h.update(str(os.path.getmtime(p)).encode())
        # Include the target CSV's modification time
        h.update(str(os.path.getmtime(self.target_csv)).encode())
        # Include config flags
        h.update(",".join(self.target_columns).encode())
        h.update(",".join(self.input_type).encode())
        h.update(str(self.keep_hs).encode())
        h.update(str(self.sanitize).encode())
        self.cache_hash = h.hexdigest()

        # 4. Point processed_dir at root/processed/<hash>
        self._proc_dir = Path(root) / "processed" / self.cache_hash

        # 5. Initialize parent with the correct root
        super().__init__(root, transform, pre_transform)

        # 6. Force-reload if requested
        if self.force_reload and os.path.exists(self.processed_paths[0]):
            os.remove(self.processed_paths[0])
            self.process()

        # 7. Load the processed data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        # We don't rely on PyG's raw folder for SDFs
        return []

    @property
    def processed_dir(self) -> str:
        return str(self._proc_dir)

    @property
    def processed_file_names(self) -> list[str]:
        return ["multi_sdf_data.pt"]

    def download(self):
        # No-op: data lives on disk already
        pass

    def process(self):
        # Ensure our processed dir exists
        os.makedirs(self.processed_dir, exist_ok=True)

        data_list = []
        for sdf_path in self.sdf_paths:
            supplier = Chem.SDMolSupplier(
                str(sdf_path),
                removeHs=not self.keep_hs,
                sanitize=self.sanitize,
            )
            mols = [
                m for m in supplier
                if m is not None
                and m.HasProp("type")
                and m.GetProp("type") in self.input_type
            ]
            if len(mols) < 2:
                print(f"Skipping {sdf_path}: need ≥2 molecules of type {self.input_type}")
                continue

            reaction_name = mols[0].GetProp("reaction")
            target_row = self.target_df.loc[self.target_df["rxn"] == reaction_name]
            if target_row.empty:
                print(f"No target for '{reaction_name}' in {sdf_path}")
                continue

            vals = [target_row[col].values[0] for col in self.target_columns]
            y = torch.tensor(vals, dtype=torch.float)

            g1 = self._mol_to_graph(mols[0])
            g2 = self._mol_to_graph(mols[1])

            pair = PairData(
                id=reaction_name,
                x_s=g1.x,
                edge_index_s=g1.edge_index,
                edge_attr_s=g1.edge_attr,
                x_t=g2.x,
                edge_index_t=g2.edge_index,
                edge_attr_t=g2.edge_attr,
                y=y.unsqueeze(0),
            )
            # tell PyG explicitly how many nodes each sub-graph has
            pair.num_nodes_s = g1.x.size(0)
            pair.num_nodes_t = g2.x.size(0)
            # (optionally) total nodes – handy for quick stats
            pair.num_nodes   = pair.num_nodes_s + pair.num_nodes_t

        # Add the pair to the list
            data_list.append(pair)


        print(f"kept   : {len(data_list)} pairs")

        if len(data_list) == 0:
            raise RuntimeError("No valid pairs – check input_type or target CSV")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _mol_to_graph(self, mol):
        atom_feats = [atom_f(a) for a in mol.GetAtoms()]
        x = torch.tensor(np.array(atom_feats), dtype=torch.float)

        conf = mol.GetConformer()
        if conf is None or not conf.Is3D():
            raise ValueError("Molecule is missing 3D coordinates")

        src, dst, edge_feats = [], [], []

        ang_feats = compute_edge_angle_features(mol)

        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()

            for u, v in ((i, j), (j, i)):
                pos_u = conf.GetAtomPosition(u)
                pos_v = conf.GetAtomPosition(v)

                vec_u = torch.tensor([pos_u.x, pos_u.y, pos_u.z], dtype=torch.float)
                vec_v = torch.tensor([pos_v.x, pos_v.y, pos_v.z], dtype=torch.float)

                rbf_uv = self._rbf(torch.norm(vec_u - vec_v))
                base_feat = torch.tensor(bond_f(b), dtype=torch.float)
                ang_feat = ang_feats.get((u, v), np.array([0.0, 0.0, -10.0]))
                full_feat = torch.cat([base_feat, rbf_uv, torch.tensor(ang_feat)], dim=0)
                
                src.append(u)
                dst.append(v)
                edge_feats.append(full_feat)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        lengths = [f.shape[0] for f in edge_feats]
        assert len(set(lengths)) == 1, f"Inconsistent edge_feat lengths: {set(lengths)}"

        edge_attr = torch.stack(edge_feats, dim=0)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


    def _rbf(self, d, D_min=0, D_max=5, D_count=10, gamma=10.0):
        D_mu = torch.linspace(D_min, D_max, D_count)
        return torch.exp(-gamma * (d - D_mu)**2)  # shape [D_count]



def compute_angle(a, b, c):
    """Compute angle (in radians) between vectors BA and BC"""
    ba = a - b
    bc = c - b
    ba /= np.linalg.norm(ba) + 1e-7
    bc /= np.linalg.norm(bc) + 1e-7
    cos_angle = np.dot(ba, bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle

def compute_edge_angle_features(mol):
    """
    Computes mean sin and cos of angles for each directed edge (i->j),
    along with a mask (-10 if undefined).
    """
    conf = mol.GetConformer()
    edge_angle_feat = {}

    for j in range(mol.GetNumAtoms()):
        nbrs = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors()]
        if len(nbrs) < 2:
            continue  # skip terminal atoms

        for i in nbrs:
            angles = []
            for k in nbrs:
                if k == i:
                    continue
                pos_i = conf.GetAtomPosition(i)
                pos_j = conf.GetAtomPosition(j)
                pos_k = conf.GetAtomPosition(k)

                angle = compute_angle(
                    np.array([pos_i.x, pos_i.y, pos_i.z]),
                    np.array([pos_j.x, pos_j.y, pos_j.z]),
                    np.array([pos_k.x, pos_k.y, pos_k.z])
                )
                angles.append(angle)

            if len(angles) > 0:
                feat = np.array([
                    np.mean(np.sin(angles)),
                    np.mean(np.cos(angles)),
                    1.0  # valid mask
                ])
            else:
                feat = np.array([0.0, 0.0, -10.0])  # invalid/missing angle

            edge_angle_feat[(i, j)] = feat

    return edge_angle_feat


import os
from pathlib import Path
from typing import List, Union, Tuple



# --- EquiData for two graphs with shared label ---
class EquiData(Data):
    """
    Holds two separate molecular graphs (source and target) plus label y and id.
    Provides num_nodes_s/num_nodes_t per example via ptrs.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Keep 'y' and 'id' as list entries when batching
        if key in ('y', 'id'):
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

    @property
    def num_nodes(self):
        return int(self.z_s.size(0) + self.z_t.size(0))

    @property
    def num_nodes_s(self):
        # Compute nodes per source graph from ptr
        ptr = self.z_s_ptr     # tensor of length num_graphs+1
        counts = ptr[1:] - ptr[:-1]
        return counts.tolist()

    @property
    def num_nodes_t(self):
        ptr = self.z_t_ptr
        counts = ptr[1:] - ptr[:-1]
        return counts.tolist()

class EquiMultiMolDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        sdf_folder: Union[str, list[str]],
        target_csv: str,
        input_type: list[str],
        target_columns: list[str],
        keep_hs: bool = False,
        sanitize: bool = True,
        transform=None,
        pre_transform=None,
        force_reload: bool = False,
    ):
        # Store config
        self.sdf_folder     = [sdf_folder] if isinstance(sdf_folder, str) else list(sdf_folder)
        self.target_csv     = target_csv
        self.input_type     = input_type
        self.target_columns = target_columns
        self.keep_hs        = keep_hs
        self.sanitize       = sanitize
        self.force_reload   = force_reload

        # Load target DataFrame
        self.target_df = pd.read_csv(self.target_csv).set_index('rxn')

        # Gather all .sdf paths
        self.sdf_paths = []
        for p_str in self.sdf_folder:
            p = Path(p_str)
            if p.is_dir():
                self.sdf_paths += sorted(p.glob('*.sdf'))
            elif p.suffix.lower() == '.sdf':
                self.sdf_paths.append(p)

        # Compute cache hash
        h = hashlib.sha1()
        for p in self.sdf_paths:
            h.update(str(p).encode())
            h.update(str(os.path.getmtime(p)).encode())
        h.update(str(os.path.getmtime(self.target_csv)).encode())
        h.update(','.join(self.target_columns).encode())
        h.update(','.join(self.input_type).encode())
        h.update(str(self.keep_hs).encode())
        h.update(str(self.sanitize).encode())
        self.cache_hash = h.hexdigest()

        super().__init__(root, transform, pre_transform)

        # Force reload if requested
        if self.force_reload and os.path.exists(self.processed_paths[0]):
            os.remove(self.processed_paths[0])
            self.process()
        
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # No raw downloads
        return []

    @property
    def processed_file_names(self):
        # Unique file per hash
        return [f'equi_multi_{self.cache_hash}.pt']

    def download(self):
        # Nothing to download
        pass


    def process(self):
        data_list = []

        # Debug: show what we're about to process
        print(f"Processing {len(self.sdf_paths)} SDF files from {self.sdf_folder}")
        print(f"RXN IDs available in CSV: {list(self.target_df.index)[:10]}{'...' if len(self.target_df)>10 else ''}")

        for sdf_path in self.sdf_paths:
            supplier = Chem.SDMolSupplier(
                str(sdf_path),
                removeHs=not self.keep_hs,
                sanitize=self.sanitize
            )
            mols = [
                m for m in supplier
                if m and m.HasProp('type') and m.GetProp('type') in self.input_type
            ]

            # Debug: what types did we see?
            types = [m.GetProp('type') for m in mols if m]
            print(f"  • {sdf_path.name} → found types {types}")

            if len(mols) < 2:
                continue

            rxn = mols[0].GetProp('reaction')
            if rxn not in self.target_df.index:
                print(f"    – skipping {sdf_path.name}: RXN '{rxn}' not in CSV")
                continue

            # Build source fragment
            conf_s = mols[0].GetConformer()
            z_s   = torch.tensor([a.GetAtomicNum() for a in mols[0].GetAtoms()], dtype=torch.long)
            pos_s = torch.tensor([[*conf_s.GetAtomPosition(i)] for i in range(mols[0].GetNumAtoms())],
                                 dtype=torch.float)

            # Build target fragment
            conf_t = mols[1].GetConformer()
            z_t   = torch.tensor([a.GetAtomicNum() for a in mols[1].GetAtoms()], dtype=torch.long)
            pos_t = torch.tensor([[*conf_t.GetAtomPosition(i)] for i in range(mols[1].GetNumAtoms())],
                                 dtype=torch.float)

            # sin/cos target (shape [2])
            sin_cos = self.target_df.loc[rxn, self.target_columns].values  # [sin, cos]
            y       = torch.tensor(sin_cos, dtype=torch.float)            # → shape [2]

            # Create Data object
            equi = EquiData(
                z_s=z_s, pos_s=pos_s,
                z_t=z_t, pos_t=pos_t,
                y=y, id=rxn
            )
            data_list.append(equi)

        # Guard against empty
        if len(data_list) == 0:
            raise RuntimeError(
                "No valid examples found in process().\n"
                f"– SDFs: {self.sdf_paths}\n"
                f"– CSV RXNs: {list(self.target_df.index)}\n"
                f"– input_type filter: {self.input_type}"
            )
        print(f"  → Built {len(data_list)} EquiData examples")
        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
# siamese_collate unchanged

def siamese_collate(batch_list):
    batch_list_s = [d for d in batch_list]
    batch_s = Batch.from_data_list(batch_list_s)
    return batch_s


class EquiDataset(InMemoryDataset):
    """
    Holds one molecular 3D coords (source) plus label y and id.
    """
    def __init__(self, root = None, 
                geoms_csv: Union[str, list[str]] = None,
                sdf_folder: Union[str, list[str]] = None,
                target_csv: str = None,
                target_columns: list[str] = None,
                transform = None, pre_transform = None, pre_filter = None, log = True, force_reload = False):
        
        # Store config
        self.geoms_csv      = geoms_csv
        self.target_csv     = target_csv
        self.target_columns = target_columns
        self.force_reload   = force_reload
        sdf_folder     = [sdf_folder] if isinstance(sdf_folder, str) else list(sdf_folder)
        # Gather all .sdf paths
        self.sdf_paths = []
        for sdf in sdf_folder:
            p = Path(sdf)
            if p.is_dir():
                self.sdf_paths += sorted(p.glob('*.sdf'))
            elif p.suffix.lower() == '.sdf':
                self.sdf_paths.append(p)

        self.df = self.read_all_sdfs()
        # Load geometry DataFrame
        if isinstance(self.geoms_csv, str):
            self.geoms_df = pd.read_csv(self.geoms_csv, index_col=False)
        elif isinstance(self.geoms_csv, list):
            self.geoms_df = pd.concat([pd.read_csv(f) for f in self.geoms_csv])
        else:
            raise ValueError("geoms_csv must be a string or a list of strings")
        self.geoms_df = self.geoms_df.set_index('rxn_id')
        # Load target DataFrame
        self.target_df = pd.read_csv(self.target_csv).set_index('rxn')


        # Compute cache hash
        h = hashlib.sha1()
        h.update(str(self.geoms_csv).encode())
        h.update(str(os.path.getmtime(self.target_csv)).encode())
        h.update(','.join(self.target_columns).encode())
        self.cache_hash = h.hexdigest()

        super().__init__(root, transform, pre_transform, force_reload=force_reload)

        # Force reload if requested
        if self.force_reload and os.path.exists(self.processed_paths[0]):
            os.remove(self.processed_paths[0])
            self.process()
        
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # No raw downloads
        return []
    
    @property
    def processed_file_names(self):
        # Unique file per hash
        return [f'equi_single_{self.cache_hash}.pt']
    
    def download(self):
        # Nothing to download
        pass

    def process(self):
        data_list = []

        for rxn_id, row in self.geoms_df.iterrows():

            # Get the geometry coords
            coords = row['coords']
            coords = ast.literal_eval(coords)
            pos = torch.tensor(coords, dtype=torch.float)
            symbols = ast.literal_eval(row['symbols'])
            z_list = [periodictable.elements.symbol(s).number for s in symbols]
            z = torch.tensor(z_list, dtype=torch.long)

            # Get the target value
            if rxn_id not in self.target_df.index:
                print(f"Skipping {rxn_id}: not found in target CSV")
                continue
            sin_cos = self.target_df.loc[rxn_id, self.target_columns].values  # [sin, cos]
            y       = torch.tensor(np.array(sin_cos), dtype=torch.float)            # → shape [2]
            # Create the data object
            tags = extract_tags(self.df, rxn_id, z)
            z = self.apply_offset(z, tags)
            data = Data(
                z=z,
                # tag_id=tags,
                pos=pos,
                y=y.unsqueeze(0),
                id=rxn_id
            )
            data_list.append(data)

        # Guard against empty
        if len(data_list) == 0:
            raise RuntimeError(
                "No valid examples found in process().\n"
                f"– CSV RXNs: {list(self.target_df.index)}\n"
            )
        print(f"  → Built {len(data_list)} EquiData examples")
        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Saved processed data to {self.processed_paths[0]}") 


    def read_all_sdfs(self):
        """
        Read all SDF files in the folder and gather mol properties from TS.
        Use Pandas Tools to read the SDF files and extract the properties.
        """
        from rdkit import Chem
        from rdkit.Chem import PandasTools
        from tqdm import tqdm

        mol_props_df = pd.DataFrame()
        for sdf_path in self.sdf_paths:
            props = PandasTools.LoadSDF(sdf_path, includeFingerprints=False)
            # Append the properties to the DataFrame
            mol_props_df = pd.concat([mol_props_df, props], ignore_index=True)
        # Remove typpes that are not 'ts'
        mol_props_df = mol_props_df[mol_props_df['type'] == 'ts']
        return mol_props_df

    def apply_offset(self, z, tag_id):
        z = z.clone()
        z[tag_id == 1] += OFFSET_ACCEPTOR
        z[tag_id == 2] += OFFSET_HYDROGEN
        z[tag_id == 3] += OFFSET_DONOR
        return z



