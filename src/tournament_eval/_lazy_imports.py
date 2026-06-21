"""Lazy heavy-dependency imports for the optional extras.

Several submodules need a heavy dependency (``numpy``, ``scipy``,
``matplotlib``) only at *call* time, not at import time, so importing the
package — and even the submodule — stays dependency-free until a function that
needs the dep is actually called.  That keeps a plain tournament run light and
surfaces a missing extra as one clear error at the point of use rather than a
top-level import failure that gates the whole submodule.

:func:`require` is the shared seam: call it at the top of any function that
needs an optional package, right before the local ``import`` that follows.  A
missing extra then surfaces as one actionable error naming the package and the
install command, rather than the bare ``ModuleNotFoundError`` a raw import
would produce.

The mechanism for *type hints* on the same submodules — referencing the heavy
package only under ``TYPE_CHECKING`` via PEP 695 ``type`` aliases, which the
runtime never evaluates — is implemented in each submodule that needs it; see
:mod:`tournament_eval.aggregation.pairwise` and
:mod:`tournament_eval.presentation.plots`.

This is a stopgap for the absence of a stdlib lazy-import (PEP 810's deferred
imports, targeting Python 3.15); until it lands, this helper is the seam.
"""

import importlib


def require(name: str, *, extra: str) -> None:
    """Raise a clear ``ImportError`` if ``name`` isn't importable, else return.

    Called at the top of any function that needs an optional heavy dependency
    (``numpy`` / ``scipy`` / ``matplotlib``), right before the local ``import``
    that follows, so a missing extra surfaces as one actionable error naming the
    package and the install command instead of a bare ``ModuleNotFoundError``.

    Parameters
    ----------
    name : str
        The top-level module to require (e.g. ``"numpy"``, ``"matplotlib"``).
    extra : str
        The project extra that provides it (e.g. ``"analysis"``, ``"plotting"``),
        named in the error message so the install hint points at the right extra
        rather than leaving the user to guess.
    """
    try:
        importlib.import_module(name)
    except ImportError as err:
        hint = (
            f"Install the {extra!r} extra: uv add 'tournament-eval[{extra}]' "
            f"(or pip install 'tournament-eval[{extra}]')."
        )
        raise ImportError(f"{name!r} is required here but isn't installed. {hint}") from err
