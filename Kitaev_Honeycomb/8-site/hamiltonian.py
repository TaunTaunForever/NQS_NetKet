from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PARENT_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "hamiltonian.py"
)

_spec = spec_from_file_location(
    "kitaev_shared_define_hamiltonian",
    PARENT_MODULE_PATH,
)
_module = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

kitaev_hamiltonian = _module.kitaev_hamiltonian
build_flux_operators = getattr(_module, "build_flux_operators", None)
