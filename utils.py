from __future__ import annotations
from typing import Iterable, Iterator
from rdkit import Chem

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    
    class StrEnum(str, Enum):
        """Fallback StrEnum for Python <3.11"""
        def __str__(self):
            return self.value

class EnumMapping(StrEnum):
    @classmethod
    def get(cls, name: str | EnumMapping) -> EnumMapping:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise KeyError(
                f"Unsupported {cls.__name__} member! got: '{name}'. expected one of: {cls.keys()}"
            )

    @classmethod
    def keys(cls) -> Iterator[str]:
        return (e.name for e in cls)

    @classmethod
    def values(cls) -> Iterator[str]:
        return (e.value for e in cls)

    @classmethod
    def items(cls) -> Iterator[tuple[str, str]]:
        return zip(cls.keys(), cls.values())


def make_mol(smi: str, keep_h: bool, add_h: bool) -> Chem.Mol:
    """build an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified
    add_h : bool
        whether to add hydrogens to the molecule

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        )
    else:
        mol = Chem.MolFromSmiles(smi)

    if mol is None:
        raise RuntimeError(f"SMILES {smi} is invalid! (RDKit returned None)")

    if add_h:
        mol = Chem.AddHs(mol)

    return mol

def make_mol_from_sdf(sdf: str, keep_h: bool, add_h: bool, mol_type: str) -> list[Chem.Mol]|Chem.Mol:
    """build an RDKit molecule from a SDF string.

    Parameters
    ----------
    sdf : str
        a SDF string.
    keep_h : bool
        whether to keep hydrogens in the input SDF. This does not add hydrogens, it only keeps them if they are specified
    add_h : bool
        whether to add hydrogens to the molecule
    mol_type : str
        the type of molecule to return. Must be one of ['all', 'ts', 'r1h', 'r2h']

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    assert mol_type in ['all', 'ts', 'r1h', 'r2h'], f"mol_type must be one of ['all', 'ts', 'r1h', 'r2h']"
    suppl = Chem.SDMolSupplier(sdf, removeHs=not keep_h)

    # Check if there are hydrogens in the molecule
    if add_h:
        for mol in suppl:
            if mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
                raise ValueError("Hydrogens are already present in the molecule")
            else:
                mol = Chem.AddHs(mol)

    if mol_type == 'all':
        mols = [mol for mol in suppl]
        return mols
    else:
        mols = [mol for mol in suppl if mol.GetProp('type') == mol_type]
        return mols[0]

def pretty_shape(shape: Iterable[int]) -> str:
    """Make a pretty string from an input shape

    Example
    --------
    >>> X = np.random.rand(10, 4)
    >>> X.shape
    (10, 4)
    >>> pretty_shape(X.shape)
    '10 x 4'
    """
    return " x ".join(map(str, shape))