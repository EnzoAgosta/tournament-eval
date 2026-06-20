"""Lazy heavy-dependency imports for the aggregation methods that need them.

The ``analysis`` extra (``numpy``/``scipy``) is optional: importing
:mod:`tournament_eval.aggregation` — and even importing a submodule like
:mod:`tournament_eval.aggregation.pairwise` — must never pull ``numpy`` into
``sys.modules``.  Only *calling* a method that needs it should.  That keeps a
plain tournament run (positional methods only) dependency-free, and surfaces a
missing extra as one clear error at the point of use rather than a top-level
import failure that gates the whole submodule.

The mechanism:

* **Type hints** — ``numpy``/``scipy`` are imported only under ``TYPE_CHECKING``
  and referenced through PEP 695 ``type`` aliases, which the runtime never
  evaluates.  So a ``@dataclass`` whose field is annotated
  ``npt.NDArray[np.int_]`` is definable without ``numpy`` installed (verified).
* **Runtime** — each function that needs ``numpy``/``scipy`` calls
  :func:`require` then does a plain local ``import``: ``require("numpy");
  import numpy as np``.  ``mypy`` understands function-local imports and types
  them correctly from the ``TYPE_CHECKING`` block, so this costs no
  ``# type: ignore`` and no ``Any``.

This is a stopgap for the absence of a stdlib lazy-import (PEP 810's deferred
imports, targeting Python 3.15); until it lands, this helper is the seam.
"""

import importlib

_INSTALL_HINT = (
    "Install the 'analysis' extra: uv add 'tournament-eval[analysis]' (or pip install 'tournament-eval[analysis]')."
)


def require(name: str) -> None:
    """Raise a clear ``ImportError`` if ``name`` isn't importable, else return.

    Called at the top of any aggregation function that needs a heavy dependency
    (``numpy`` / ``scipy``), right before the local ``import`` that follows.  A
    missing extra then surfaces as one actionable error naming the package and
    the install command, rather than the bare ``ModuleNotFoundError`` a raw
    ``import numpy`` would produce.

    Parameters
    ----------
    name : str
        The top-level module to require (e.g. ``"numpy"``).
    """
    try:
        importlib.import_module(name)
    except ImportError as err:
        raise ImportError(f"{name!r} is required here but isn't installed. {_INSTALL_HINT}") from err
