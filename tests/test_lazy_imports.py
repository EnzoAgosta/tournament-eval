import importlib
from unittest import mock

import pytest

import tournament_eval
from tournament_eval import _OPTIONAL_CLIENTS


@pytest.mark.parametrize("name", list(_OPTIONAL_CLIENTS))
def test_optional_client_resolves_lazily(name: str) -> None:
    module_name = _OPTIONAL_CLIENTS[name][0]
    module = importlib.import_module(module_name)
    assert getattr(tournament_eval, name) is getattr(module, name)


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'Nonexistent'"):
        _ = tournament_eval.Nonexistent


def test_missing_extra_raises_install_hint() -> None:
    with (
        mock.patch("tournament_eval.importlib.import_module", side_effect=ImportError("no module")),
        pytest.raises(ImportError, match="OpenAIClient requires the 'openai' extra"),
    ):
        _ = tournament_eval.OpenAIClient
