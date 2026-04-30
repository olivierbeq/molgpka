from enum import Enum
from typing import TypedDict, Dict, List, Any

from rdkit import Chem


class BackendType(str, Enum):
    MOLGPKA = "molgpka"
    PKALEARN = "pkalearn"


class LadderStep(TypedDict):
    smiles: str
    center: int
    pka: float


class MicrostateResult(TypedDict):
    major_state: Chem.Mol
    pka: float | None
    ladder: List[LadderStep] | Dict[str, Any]
