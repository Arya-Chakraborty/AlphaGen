from __future__ import annotations

from random import Random

from molecule_gen.generate.chemistry import is_reasonable_molecule


def _mol_from_smiles(smiles: str):
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def _canon(rw) -> str | None:
    from rdkit import Chem

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    smiles = Chem.MolToSmiles(mol, canonical=True)
    if not smiles or not is_reasonable_molecule(smiles):
        return None
    return smiles


def add_methyl(smiles: str, rng: Random) -> str | None:
    from rdkit import Chem

    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    carbons = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0
    ]
    if not carbons:
        return None
    rng.shuffle(carbons)
    for idx in carbons:
        rw = Chem.RWMol(mol)
        new_idx = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(idx, new_idx, Chem.BondType.SINGLE)
        result = _canon(rw)
        if result and result != smiles:
            return result
    return None


def remove_methyl(smiles: str, rng: Random) -> str | None:
    from rdkit import Chem

    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    terminals = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetDegree() != 1 or atom.IsInRing():
            continue
        if atom.GetTotalNumHs() < 2:
            continue
        terminals.append(atom.GetIdx())
    if not terminals:
        return None
    rng.shuffle(terminals)
    for idx in terminals:
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(idx)
        result = _canon(rw)
        if result and result != smiles:
            return result
    return None


def swap_hydrogen_fluorine(smiles: str, rng: Random) -> str | None:
    from rdkit import Chem

    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    fluorines = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9]
    carbons = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0
    ]
    ops: list[str] = []
    if fluorines:
        ops.append("def")
    if carbons:
        ops.append("add")
    if not ops:
        return None
    rng.shuffle(ops)
    for op in ops:
        rw = Chem.RWMol(mol)
        if op == "def":
            rng.shuffle(fluorines)
            rw.RemoveAtom(fluorines[0])
        else:
            rng.shuffle(carbons)
            new_idx = rw.AddAtom(Chem.Atom(9))
            rw.AddBond(carbons[0], new_idx, Chem.BondType.SINGLE)
        result = _canon(rw)
        if result and result != smiles:
            return result
    return None


def insert_ch2(smiles: str, rng: Random) -> str | None:
    from rdkit import Chem

    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetAtomicNum() != 6 or b.GetAtomicNum() != 6:
            continue
        if a.GetIsAromatic() or b.GetIsAromatic():
            continue
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    if not bonds:
        return None
    rng.shuffle(bonds)
    for begin, end in bonds:
        rw = Chem.RWMol(mol)
        rw.RemoveBond(begin, end)
        mid = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(begin, mid, Chem.BondType.SINGLE)
        rw.AddBond(mid, end, Chem.BondType.SINGLE)
        result = _canon(rw)
        if result and result != smiles:
            return result
    return None


def remove_ch2(smiles: str, rng: Random) -> str | None:
    from rdkit import Chem

    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    mids = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetDegree() != 2 or atom.IsInRing():
            continue
        if atom.GetIsAromatic() or atom.GetTotalNumHs() < 1:
            continue
        nbrs = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        if len(nbrs) != 2:
            continue
        n1, n2 = mol.GetAtomWithIdx(nbrs[0]), mol.GetAtomWithIdx(nbrs[1])
        if n1.GetAtomicNum() != 6 or n2.GetAtomicNum() != 6:
            continue
        mids.append((atom.GetIdx(), nbrs[0], nbrs[1]))
    if not mids:
        return None
    rng.shuffle(mids)
    for idx, n1, n2 in mids:
        rw = Chem.RWMol(mol)
        if rw.GetBondBetweenAtoms(n1, n2) is None:
            rw.AddBond(n1, n2, Chem.BondType.SINGLE)
        rw.RemoveAtom(idx)
        result = _canon(rw)
        if result and result != smiles:
            return result
    return None


_OPS = (
    add_methyl,
    remove_methyl,
    swap_hydrogen_fluorine,
    insert_ch2,
    remove_ch2,
)


def local_edit_smiles(smiles: str, rng: Random) -> str | None:
    """One tiny graph edit. None if nothing valid was produced."""
    order = list(_OPS)
    rng.shuffle(order)
    for op in order:
        result = op(smiles, rng)
        if result and result != smiles:
            return result
    return None
