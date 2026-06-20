"""Tests for the ranking template — aliasing, prompt rendering, and the dynamic
alias check (the static shape is now validated by pydantic-ai at the provider, so
it isn't tested here)."""

import pytest
from pydantic import BaseModel

from tests.conftest import GenerationResultFactory, RankingTaskFactory
from tournament_eval.ranking import (
    DefaultRankingTemplate,
    ParsedRanking,
    RankingResponse,
    alias_for_index,
)


def test_alias_for_index_first_letter() -> None:
    assert alias_for_index(0) == "A"


def test_alias_for_index_last_single_letter() -> None:
    assert alias_for_index(25) == "Z"


def test_alias_for_index_rolls_over_to_two_letters() -> None:
    assert alias_for_index(26) == "AA"
    assert alias_for_index(27) == "AB"
    assert alias_for_index(51) == "AZ"
    assert alias_for_index(52) == "BA"


def test_alias_for_index_last_two_letters_rolls_over_to_three() -> None:
    assert alias_for_index(701) == "ZZ"
    assert alias_for_index(702) == "AAA"


def test_response_model_is_ranking_response() -> None:
    assert DefaultRankingTemplate().response_model is RankingResponse


def test_render_includes_task_prompt_ranking_prompt_and_candidates(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    ranking_task = make_ranking_task(ranking_prompt="Rank by clarity.")
    candidates = {
        "A": make_generation_result(generation_prompt="Write a haiku.", output="output-a"),
        "B": make_generation_result(generation_prompt="Write a haiku.", output="output-b"),
    }
    template = DefaultRankingTemplate()

    prompt = template.render(ranking_task, candidates)

    assert "The models were given this task:\nWrite a haiku." in prompt
    assert "Rank by clarity." in prompt
    assert "NO TIES" in prompt
    assert "Candidates:" in prompt
    assert "A: output-a" in prompt
    assert "B: output-b" in prompt


def test_render_states_candidate_count_and_enumerates_aliases(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    candidates = {
        alias: make_generation_result(generation_prompt="p", output=f"o-{alias}") for alias in ["A", "B", "C", "D"]
    }
    template = DefaultRankingTemplate()

    prompt = template.render(make_ranking_task(), candidates)

    # Count and alias enumeration up front — the lever that cut missing-alias
    # failures to zero in the live Ollama experiment.
    assert "There are 4 candidates, labelled: A, B, C, D." in prompt
    assert "exactly once" in prompt
    assert "no omissions, no duplicates, no extras" in prompt


def test_render_example_uses_the_actual_aliases(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    candidates = {
        alias: make_generation_result(generation_prompt="p", output=f"o-{alias}") for alias in ["A", "B", "C"]
    }
    template = DefaultRankingTemplate()

    prompt = template.render(make_ranking_task(), candidates)

    # The format example reflects the real candidate set rather than a hardcoded
    # ["A", "B", "C"] that misleads when the field size differs.
    assert '["A", "B", "C"]' in prompt


def test_render_is_mechanism_neutral(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    candidates = {"A": make_generation_result(output="o-a"), "B": make_generation_result(output="o-b")}
    template = DefaultRankingTemplate()

    prompt = template.render(make_ranking_task(), candidates)

    # pydantic-ai delivers structured output via tool-calling, not raw JSON; the
    # prompt must not assert a delivery mechanism that could mismatch and confuse
    # weaker models.
    assert "Respond ONLY with a JSON object" not in prompt
    assert "Return two fields:" in prompt
    assert "`ranking`" in prompt
    assert "`reasoning`" in prompt


def test_render_raises_on_no_candidates(make_ranking_task: RankingTaskFactory) -> None:
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match="render needs at least one candidate"):
        template.render(make_ranking_task(), {})


def test_render_raises_when_candidates_have_different_generation_prompts(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    candidates = {
        "A": make_generation_result(generation_prompt="Write a haiku."),
        "B": make_generation_result(generation_prompt="Write a limerick."),
    }
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match="All candidates must have the same generation_prompt"):
        template.render(make_ranking_task(), candidates)


def test_render_omits_reasoning_by_default(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    candidates = {"A": make_generation_result(output="answer", reasoning="my secret reasoning")}
    template = DefaultRankingTemplate()

    prompt = template.render(make_ranking_task(), candidates)

    assert "A: answer" in prompt
    assert "my secret reasoning" not in prompt
    assert "A reasoning:" not in prompt


def test_render_includes_reasoning_when_enabled(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    candidates = {"A": make_generation_result(output="answer", reasoning="my reasoning")}
    template = DefaultRankingTemplate(include_reasoning=True)

    prompt = template.render(make_ranking_task(), candidates)

    assert "A: answer" in prompt
    assert "A reasoning: my reasoning" in prompt


def test_render_skips_reasoning_line_when_candidate_has_none_even_if_enabled(
    make_ranking_task: RankingTaskFactory, make_generation_result: GenerationResultFactory
) -> None:
    candidates = {"A": make_generation_result(output="answer", reasoning=None)}
    template = DefaultRankingTemplate(include_reasoning=True)

    prompt = template.render(make_ranking_task(), candidates)

    assert "A: answer" in prompt
    assert "A reasoning:" not in prompt


def test_parse_returns_ranking_and_reasoning() -> None:
    template = DefaultRankingTemplate()

    parsed = template.parse(RankingResponse(ranking=["B", "A"], reasoning="B reads better"), {"A", "B"})

    assert isinstance(parsed, ParsedRanking)
    assert parsed.ranking == ["B", "A"]
    assert parsed.reasoning == "B reads better"


def test_parse_reasoning_is_none_when_absent() -> None:
    template = DefaultRankingTemplate()

    parsed = template.parse(RankingResponse(ranking=["A"], reasoning=None), {"A"})

    assert parsed.ranking == ["A"]
    assert parsed.reasoning is None


def test_parse_raises_on_wrong_model_type() -> None:
    class Other(BaseModel):
        x: int

    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match="Expected a RankingResponse"):
        template.parse(Other(x=1), {"A"})


def test_parse_raises_on_unknown_alias() -> None:
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match="Unknown alias in ranking: 'Z'"):
        template.parse(RankingResponse(ranking=["Z"]), {"A"})


def test_parse_raises_on_duplicate_alias() -> None:
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match="Duplicate alias in ranking: 'A'"):
        template.parse(RankingResponse(ranking=["A", "A"]), {"A", "B"})


def test_parse_raises_when_aliases_are_missing() -> None:
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match=r"Missing aliases in ranking: \['B'\]"):
        template.parse(RankingResponse(ranking=["A"]), {"A", "B"})


def test_ranking_response_docstring_is_directive() -> None:
    # pydantic-ai surfaces the response model's class docstring as the tool
    # description, so it should reinforce the no-ties / no-omissions rule at the
    # schema layer rather than just describe the type.
    assert "strict total order" in RankingResponse.__doc__
    assert "no ties" in RankingResponse.__doc__
    assert "no omissions" in RankingResponse.__doc__


def test_ranking_response_field_descriptions_enforce_completeness() -> None:
    # Field descriptions are injected into the tool schema sent to every provider,
    # so the ranking field should carry the every-alias-exactly-once rule.
    fields = RankingResponse.model_fields
    assert "exactly once" in fields["ranking"].description
    assert "no omissions" in fields["ranking"].description
    assert fields["reasoning"].description is not None
