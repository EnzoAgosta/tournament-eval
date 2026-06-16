import pytest

from tests.conftest import GenerationResultFactory, RankingTaskFactory
from tournament_eval.ranking import (
    DefaultRankingTemplate,
    ParsedRanking,
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
    assert "NO TIES ARE ALLOWED" in prompt
    assert "Candidates:" in prompt
    assert "A: output-a" in prompt
    assert "B: output-b" in prompt


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


def test_schema_constrains_response_to_a_ranking_object() -> None:
    schema = DefaultRankingTemplate().schema

    assert schema["type"] == "object"
    assert schema["required"] == ["ranking"]
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert properties["ranking"]["type"] == "array"
    assert properties["reasoning"]["type"] == "string"


def test_parse_returns_ranking_and_reasoning() -> None:
    template = DefaultRankingTemplate()

    parsed = template.parse({"ranking": ["B", "A"], "reasoning": "B reads better"}, {"A", "B"})

    assert isinstance(parsed, ParsedRanking)
    assert parsed.ranking == ["B", "A"]
    assert parsed.reasoning == "B reads better"


def test_parse_reasoning_is_none_when_absent() -> None:
    template = DefaultRankingTemplate()

    parsed = template.parse({"ranking": ["A"]}, {"A"})

    assert parsed.ranking == ["A"]
    assert parsed.reasoning is None


def test_parse_raises_when_data_is_not_a_dict() -> None:
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match="Expected JSON object, got list"):
        template.parse(["A"], {"A"})  # type: ignore[arg-type]


def test_parse_raises_when_ranking_is_not_a_list() -> None:
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match='"ranking" must be a list, got str'):
        template.parse({"ranking": "A"}, {"A"})


def test_parse_raises_when_ranking_entry_is_not_a_string() -> None:
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match="Ranking entry must be a string, got int"):
        template.parse({"ranking": [1]}, {"A"})


def test_parse_raises_on_unknown_alias() -> None:
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match="Unknown alias in ranking: 'Z'"):
        template.parse({"ranking": ["Z"]}, {"A"})


def test_parse_raises_on_duplicate_alias() -> None:
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match="Duplicate alias in ranking: 'A'"):
        template.parse({"ranking": ["A", "A"]}, {"A", "B"})


def test_parse_raises_when_aliases_are_missing() -> None:
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match=r"Missing aliases in ranking: \['B'\]"):
        template.parse({"ranking": ["A"]}, {"A", "B"})


def test_parse_raises_when_reasoning_is_not_a_string() -> None:
    template = DefaultRankingTemplate()
    with pytest.raises(ValueError, match='"reasoning" must be a string, got int'):
        template.parse({"ranking": ["A"], "reasoning": 1}, {"A"})


def test_parse_accepts_explicit_null_reasoning() -> None:
    template = DefaultRankingTemplate()

    parsed = template.parse({"ranking": ["A"], "reasoning": None}, {"A"})

    assert parsed.ranking == ["A"]
    assert parsed.reasoning is None
