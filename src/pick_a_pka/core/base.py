from abc import ABC, abstractmethod

from rdkit import Chem

class BasePKaModel(ABC):
    def __init__(self, device="cpu"):
        self.device = device

    @abstractmethod
    def predict_pka(self, mol: Chem.Mol | list[Chem.Mol]) -> list[dict[int, float]]:
        pass

    def predict_microstates(self, mol: Chem.Mol | list[Chem.Mol], ph: float | list[float] = 7.4,
                            ph_range: tuple = None, ph_step: float = None
                            ) -> list[dict[float, Chem.Mol] | dict[float, dict[float, Chem.Mol]]]:
        raise NotImplementedError

    def dispose(self):
        pass
