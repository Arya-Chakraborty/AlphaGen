from __future__ import annotations

# Neutral C/N/O/S/P/halogen + branching/rings. No boron, no formal charges.
_ALLOWED_MUTATION_TOKENS = (
    "[C]",
    "[=C]",
    "[#C]",
    "[N]",
    "[=N]",
    "[#N]",
    "[O]",
    "[=O]",
    "[S]",
    "[=S]",
    "[#S]",
    "[P]",
    "[=P]",
    "[#P]",
    "[F]",
    "[Cl]",
    "[Br]",
    "[I]",
    "[H]",
    "[Branch1]",
    "[Branch2]",
    "[Branch3]",
    "[=Branch1]",
    "[=Branch2]",
    "[=Branch3]",
    "[#Branch1]",
    "[#Branch2]",
    "[#Branch3]",
    "[Ring1]",
    "[Ring2]",
    "[Ring3]",
    "[=Ring1]",
    "[=Ring2]",
    "[=Ring3]",
)

ALLOWED_ATOMIC_NUMS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})
MIN_HEAVY_ATOMS = 6
MAX_HEAVY_ATOMS = 80


def mutation_alphabet() -> tuple[str, ...]:
    """SELFIES tokens the GA may insert or substitute."""
    from molecule_gen.generate.selfies_util import robust_alphabet

    allowed = set(_ALLOWED_MUTATION_TOKENS)
    return tuple(token for token in robust_alphabet() if token in allowed)


def is_reasonable_molecule(smiles: str | None) -> bool:
    """Conservative organic filter. Used on mutants (and GPT seeds), not as a corpus check."""
    if not smiles:
        return False
    from rdkit import Chem, RDLogger

    from molecule_gen.generate.plausibility import unstable_motifs_in_mol

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    if unstable_motifs_in_mol(mol):
        return False
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < MIN_HEAVY_ATOMS or n_heavy > MAX_HEAVY_ATOMS:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in ALLOWED_ATOMIC_NUMS:
            return False
        if atom.GetFormalCharge() != 0:
            return False
        if atom.GetNumRadicalElectrons():
            return False
        if atom.GetAtomicNum() == 6:
            n_double = sum(
                1
                for bond in atom.GetBonds()
                if bond.GetBondType() == Chem.BondType.DOUBLE
            )
            if n_double >= 2:
                return False
    return True
