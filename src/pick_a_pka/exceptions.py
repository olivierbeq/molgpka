class PickAPkaError(Exception):
    """Base exception for pick-a-pka package."""
    pass


class InvalidBackendError(PickAPkaError):
    """Raised when an unknown backend is requested."""
    pass


class InvalidMoleculeError(PickAPkaError):
    """Raised when a SMILES string or RDKit molecule is invalid."""
    pass


class ResourceNotFoundError(PickAPkaError):
    """Raised when model weights or reference files cannot be found."""
    pass
