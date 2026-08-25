"""
Phase 11 test suite: .env loading, and the Ollama / Anthropic teacher
clients + Ollama model lifecycle manager (mocked -- no real network/process
calls in the automated suite; see scripts/generate_teacher_data.py and the
real-run notes in PLAN.md for actual end-to-end validation).

Run with: pytest tests/test_teachers.py -v
"""
import json
import subprocess
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teachers.base import Teacher  # noqa: E402
from teachers.env import get_env, load_env, parse_env_file, require_env  # noqa: E402
from teachers.ollama_client import OllamaError, generate as ollama_generate  # noqa: E402
from teachers.ollama_manager import (  # noqa: E402
    OllamaManagerError,
    delete_model,
    is_present,
    list_models,
    pull_model,
)
from teachers.roster import build_roster, capabilities_for, teachers_for_category  # noqa: E402
from teachers.trajectory import Trajectory, compute_disagreement_scores  # noqa: E402


# --------------------------------------------------------------------------
# .env parsing/loading
# --------------------------------------------------------------------------


def test_parse_env_file_basic():
    text = "FOO=bar\nBAZ=qux\n"
    assert parse_env_file(text) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_env_file_skips_comments_and_blank_lines():
    text = "# comment\n\nFOO=bar\n  # indented comment\nBAZ=qux\n\n"
    assert parse_env_file(text) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_env_file_strips_quotes():
    text = 'FOO="bar baz"\nQUX=\'single\'\n'
    assert parse_env_file(text) == {"FOO": "bar baz", "QUX": "single"}


def test_parse_env_file_ignores_lines_without_equals():
    text = "FOO=bar\nnotanassignment\nBAZ=qux\n"
    assert parse_env_file(text) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_env_file_empty_value():
    assert parse_env_file("FOO=\n") == {"FOO": ""}


def test_load_env_missing_file_returns_empty_dict(tmp_path):
    result = load_env(str(tmp_path / "does_not_exist.env"))
    assert result == {}


def test_load_env_sets_os_environ_without_overriding_existing(tmp_path, monkeypatch):
    env_file = tmp_path / "test.env"
    env_file.write_text("NEURAQUILL_TEST_A=from_file\nNEURAQUILL_TEST_B=from_file\n")

    monkeypatch.delenv("NEURAQUILL_TEST_A", raising=False)
    monkeypatch.setenv("NEURAQUILL_TEST_B", "from_shell")

    load_env(str(env_file))

    import os

    assert os.environ["NEURAQUILL_TEST_A"] == "from_file"
    assert os.environ["NEURAQUILL_TEST_B"] == "from_shell"  # real env wins


def test_require_env_raises_clear_error_when_unset(monkeypatch):
    monkeypatch.delenv("NEURAQUILL_TEST_MISSING", raising=False)
    with pytest.raises(RuntimeError, match="NEURAQUILL_TEST_MISSING"):
        require_env("NEURAQUILL_TEST_MISSING")


def test_require_env_raises_on_empty_string(monkeypatch):
    monkeypatch.setenv("NEURAQUILL_TEST_EMPTY", "")
    with pytest.raises(RuntimeError):
        require_env("NEURAQUILL_TEST_EMPTY")


def test_get_env_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("NEURAQUILL_TEST_DEFAULT", raising=False)
    assert get_env("NEURAQUILL_TEST_DEFAULT", "fallback") == "fallback"


# --------------------------------------------------------------------------
# Ollama HTTP client (mocked urlopen)
# --------------------------------------------------------------------------


def _fake_response(body: dict):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(body).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


def test_ollama_generate_returns_response_text():
    with patch("teachers.ollama_client.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = _fake_response({"response": "hello from ollama"})
        result = ollama_generate("say hi", model="qwen2.5:1.5b")
        assert result == "hello from ollama"


def test_ollama_generate_sends_correct_payload():
    with patch("teachers.ollama_client.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = _fake_response({"response": "ok"})
        ollama_generate("my prompt", model="gemma2:2b", system="be terse", temperature=0.2)

        sent_request = mock_urlopen.call_args[0][0]
        payload = json.loads(sent_request.data)
        assert payload["model"] == "gemma2:2b"
        assert payload["prompt"] == "my prompt"
        assert payload["system"] == "be terse"
        assert payload["options"]["temperature"] == 0.2
        assert payload["stream"] is False


def test_ollama_generate_connection_error_raises_ollama_error():
    with patch("teachers.ollama_client.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        with pytest.raises(OllamaError, match="could not reach Ollama"):
            ollama_generate("hi", model="qwen2.5:1.5b")


def test_ollama_generate_missing_response_field_raises():
    with patch("teachers.ollama_client.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = _fake_response({"unexpected": "shape"})
        with pytest.raises(OllamaError, match="missing 'response'"):
            ollama_generate("hi", model="qwen2.5:1.5b")


# --------------------------------------------------------------------------
# Ollama model lifecycle manager (mocked subprocess)
# --------------------------------------------------------------------------


def _fake_completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_list_models_parses_output():
    fake_output = (
        "NAME                SIZE      MODIFIED\n"
        "gemma2:2b           1.6 GB    6 months ago\n"
        "qwen2.5:1.5b        986 MB    4 months ago\n"
    )
    with patch("teachers.ollama_manager.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(stdout=fake_output)
        models = list_models()
        assert models == ["gemma2:2b", "qwen2.5:1.5b"]


def test_list_models_raises_on_failure():
    with patch("teachers.ollama_manager.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(stderr="ollama not found", returncode=1)
        with pytest.raises(OllamaManagerError):
            list_models()


def test_is_present_true_and_false():
    with patch("teachers.ollama_manager.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(stdout="NAME  SIZE\ngemma2:2b  1.6GB\n")
        assert is_present("gemma2:2b") is True
        assert is_present("nonexistent:1b") is False


def test_pull_model_calls_ollama_pull():
    with patch("teachers.ollama_manager.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed()
        pull_model("mistral:7b")
        args = mock_run.call_args[0][0]
        assert args == ["ollama", "pull", "mistral:7b"]
        # progress-bar stdout is discarded, not captured (a multi-GB pull's
        # redrawing progress bar would otherwise dump hundreds of KB of noise)
        assert mock_run.call_args.kwargs["stdout"] == subprocess.DEVNULL


def test_pull_model_raises_on_failure():
    with patch("teachers.ollama_manager.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(stderr="pull failed", returncode=1)
        with pytest.raises(OllamaManagerError, match="pull failed"):
            pull_model("nonexistent-model:1b", retries=1)  # no retries -> no real sleep in the test


def test_pull_model_retries_on_transient_failure_then_succeeds():
    with patch("teachers.ollama_manager.subprocess.run") as mock_run, \
         patch("teachers.ollama_manager.time.sleep") as mock_sleep:
        mock_run.side_effect = [
            _fake_completed(stderr="network blip", returncode=1),
            _fake_completed(returncode=0),
        ]
        pull_model("mistral:latest", retries=3)
        assert mock_run.call_count == 2
        mock_sleep.assert_called_once()  # backed off once, between the two attempts


def test_pull_model_catches_timeout_expired_and_retries():
    """Regression test: subprocess.TimeoutExpired is raised BY
    subprocess.run itself (not a returncode failure), so it has to be
    caught explicitly or it propagates straight out of pull_model
    uncaught -- which is exactly what happened in the real Phase 11 run
    (a 30-minute stall on mistral:latest crashed the whole script,
    losing already-collected results from three other teachers)."""
    with patch("teachers.ollama_manager.subprocess.run") as mock_run, \
         patch("teachers.ollama_manager.time.sleep"):
        mock_run.side_effect = [
            subprocess.TimeoutExpired(cmd=["ollama", "pull", "mistral:latest"], timeout=1800),
            _fake_completed(returncode=0),
        ]
        pull_model("mistral:latest", retries=3)  # must not raise
        assert mock_run.call_count == 2


def test_pull_model_all_timeouts_raises_ollama_manager_error_not_raw_timeout():
    with patch("teachers.ollama_manager.subprocess.run") as mock_run, \
         patch("teachers.ollama_manager.time.sleep"):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ollama", "pull", "x"], timeout=1800)
        with pytest.raises(OllamaManagerError, match="after 3 attempts"):
            pull_model("mistral:latest", retries=3)


def test_pull_model_gives_up_after_all_retries_exhausted():
    with patch("teachers.ollama_manager.subprocess.run") as mock_run, \
         patch("teachers.ollama_manager.time.sleep"):
        mock_run.return_value = _fake_completed(stderr="still failing", returncode=1)
        with pytest.raises(OllamaManagerError, match="after 3 attempts"):
            pull_model("mistral:latest", retries=3)
        assert mock_run.call_count == 3


def test_delete_model_calls_ollama_rm():
    with patch("teachers.ollama_manager.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed()
        delete_model("mistral:7b")
        args = mock_run.call_args[0][0]
        assert args == ["ollama", "rm", "mistral:7b"]


def test_delete_model_raises_on_failure():
    with patch("teachers.ollama_manager.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(stderr="not found", returncode=1)
        with pytest.raises(OllamaManagerError):
            delete_model("nonexistent-model:1b")


# --------------------------------------------------------------------------
# Anthropic teacher client (mocked SDK -- no real API key or spend needed)
# --------------------------------------------------------------------------

import teachers.anthropic_client as anthropic_client_module  # noqa: E402
from teachers.spend_tracker import BudgetExceededError, SpendTracker  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_anthropic_tracker(monkeypatch, tmp_path):
    """Every test in this file gets its own fresh, isolated SpendTracker
    (a tmp log file) instead of the module's lazily-initialized global --
    so tests never leak spend state into each other or touch the real
    data/anthropic_spend_log.json."""
    monkeypatch.setattr(
        anthropic_client_module, "_tracker",
        SpendTracker(budget_usd=10.0, log_path=str(tmp_path / "spend_test.json")),
    )
    yield


def _fake_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _fake_claude_response(text, input_tokens=100, output_tokens=200):
    response = MagicMock()
    response.content = [_fake_text_block(text)]
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


def test_anthropic_generate_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from teachers.anthropic_client import generate as claude_generate

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        claude_generate("hello")


def test_anthropic_generate_extracts_text_blocks(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    from teachers.anthropic_client import generate as claude_generate

    with patch("teachers.anthropic_client.anthropic.Anthropic") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_claude_response("hello world")
        mock_client_cls.return_value = mock_client

        result = claude_generate("say hi", model="claude-haiku-4-5")
        assert result == "hello world"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5"
        assert call_kwargs["messages"] == [{"role": "user", "content": "say hi"}]


def test_anthropic_generate_passes_system_prompt(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    from teachers.anthropic_client import generate as claude_generate

    with patch("teachers.anthropic_client.anthropic.Anthropic") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_claude_response("ok")
        mock_client_cls.return_value = mock_client

        claude_generate("hi", system="be terse", model="claude-haiku-4-5")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "be terse"


# --------------------------------------------------------------------------
# spend tracking + hard budget enforcement -- the safety-critical part
# --------------------------------------------------------------------------


def test_estimate_cost_known_model():
    from teachers.spend_tracker import estimate_cost

    # claude-opus-5: $5/1M input, $25/1M output
    cost = estimate_cost("claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(30.0)


def test_estimate_cost_unknown_model_uses_most_expensive_rate_not_zero():
    from teachers.spend_tracker import estimate_cost

    cost_unknown = estimate_cost("some-future-model", input_tokens=1_000_000, output_tokens=1_000_000)
    cost_opus = estimate_cost("claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_unknown >= cost_opus  # fail safe: never under-count an unrecognized model's cost


def test_spend_tracker_persists_across_instances(tmp_path):
    log_path = str(tmp_path / "spend.json")
    tracker1 = SpendTracker(budget_usd=10.0, log_path=log_path)
    tracker1.record("claude-opus-5", input_tokens=100_000, output_tokens=10_000)
    spent_after_first = tracker1.total_spent
    assert spent_after_first > 0

    tracker2 = SpendTracker(budget_usd=10.0, log_path=log_path)  # simulates a fresh process/run
    assert tracker2.total_spent == pytest.approx(spent_after_first)


def test_spend_tracker_check_budget_raises_before_exceeding(tmp_path):
    tracker = SpendTracker(budget_usd=1.0, log_path=str(tmp_path / "spend.json"))
    tracker.record("claude-opus-5", input_tokens=100_000, output_tokens=10_000)  # ~$0.75
    with pytest.raises(BudgetExceededError):
        tracker.check_budget(estimated_next_cost=0.30)  # would push total past $1.00


def test_spend_tracker_check_budget_allows_when_within_budget(tmp_path):
    tracker = SpendTracker(budget_usd=10.0, log_path=str(tmp_path / "spend.json"))
    tracker.check_budget(estimated_next_cost=1.0)  # must not raise


def test_spend_tracker_remaining():
    tracker = SpendTracker(budget_usd=10.0, log_path="unused")
    tracker._total_spent = 3.0  # simulate prior spend without touching disk
    assert tracker.remaining == pytest.approx(7.0)


def test_anthropic_generate_refuses_call_over_budget_without_hitting_api(monkeypatch, tmp_path):
    """The core safety guarantee: a call whose worst-case cost would
    exceed the remaining budget must be refused BEFORE the API is ever
    called -- not logged as an overspend after the fact."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(
        anthropic_client_module, "_tracker",
        SpendTracker(budget_usd=0.001, log_path=str(tmp_path / "tiny_budget.json")),  # essentially $0
    )
    from teachers.anthropic_client import generate as claude_generate

    with patch("teachers.anthropic_client.anthropic.Anthropic") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        with pytest.raises(BudgetExceededError):
            claude_generate("a very long prompt " * 100, max_tokens=2000)

        mock_client.messages.create.assert_not_called()  # the network call must never happen


def test_anthropic_generate_records_actual_spend_after_call(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    from teachers.anthropic_client import generate as claude_generate

    with patch("teachers.anthropic_client.anthropic.Anthropic") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_claude_response("ok", input_tokens=500, output_tokens=800)
        mock_client_cls.return_value = mock_client

        claude_generate("hi")
        tracker = anthropic_client_module._tracker
        assert tracker.total_spent > 0
        assert len(tracker._calls) == 1
        assert tracker._calls[0]["input_tokens"] == 500
        assert tracker._calls[0]["output_tokens"] == 800


def test_anthropic_generate_spend_accumulates_across_multiple_calls(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    from teachers.anthropic_client import generate as claude_generate

    with patch("teachers.anthropic_client.anthropic.Anthropic") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_claude_response("ok", input_tokens=1000, output_tokens=1000)
        mock_client_cls.return_value = mock_client

        claude_generate("hi")
        spent_after_one = anthropic_client_module._tracker.total_spent
        claude_generate("hi again")
        spent_after_two = anthropic_client_module._tracker.total_spent

        assert spent_after_two == pytest.approx(spent_after_one * 2)


# --------------------------------------------------------------------------
# Teacher (common interface over providers)
# --------------------------------------------------------------------------


def test_teacher_dispatches_to_ollama():
    teacher = Teacher(name="ollama/gemma3:4b", provider="ollama", model="gemma3:4b", capabilities=["general"])
    with patch("teachers.base.ollama_client.generate") as mock_gen:
        mock_gen.return_value = "ollama reply"
        result = teacher.generate("hello", system="sys", temperature=0.5)
        assert result == "ollama reply"
        _, kwargs = mock_gen.call_args
        assert kwargs["model"] == "gemma3:4b"
        assert kwargs["system"] == "sys"
        assert kwargs["temperature"] == 0.5


def test_teacher_dispatches_to_anthropic():
    teacher = Teacher(name="anthropic/claude-opus-5", provider="anthropic", model="claude-opus-5", capabilities=["math"])
    with patch("teachers.base.anthropic_client.generate") as mock_gen:
        mock_gen.return_value = "claude reply"
        result = teacher.generate("hello", system="sys")
        assert result == "claude reply"
        _, kwargs = mock_gen.call_args
        assert kwargs["model"] == "claude-opus-5"
        assert kwargs["system"] == "sys"


def test_teacher_unknown_provider_raises():
    teacher = Teacher(name="x/y", provider="bogus", model="y", capabilities=[])
    with pytest.raises(ValueError, match="unknown teacher provider"):
        teacher.generate("hello")


# --------------------------------------------------------------------------
# roster: capability profiles + building from env + category filtering
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected_contains",
    [
        ("deepseek-r1:8b", "math"),
        ("qwen3.5:9b", "code"),
        ("gemma3:4b", "reasoning"),
        ("llama3.2:latest", "reasoning"),
        ("mistral:latest", "reasoning"),
        ("glm4:9b", "math"),
        ("command-r7b:latest", "research"),
        ("claude-opus-5", "research"),
    ],
)
def test_capabilities_for_known_families(model, expected_contains):
    assert expected_contains in capabilities_for(model)


def test_capabilities_for_is_case_insensitive():
    assert capabilities_for("DeepSeek-R1:8b") == capabilities_for("deepseek-r1:8b")


def test_capabilities_for_unknown_model_gets_default():
    assert capabilities_for("some-random-model:1b") == ["general"]


def test_build_roster_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_TEACHER_MODELS", "gemma3:4b,qwen3.5:9b")
    monkeypatch.setenv("CLAUDE_TEACHER_MODEL", "claude-opus-5")
    roster = build_roster()
    names = [t.name for t in roster]
    assert names == ["ollama/gemma3:4b", "ollama/qwen3.5:9b", "anthropic/claude-opus-5"]
    assert roster[0].provider == "ollama"
    assert roster[0].model == "gemma3:4b"
    assert roster[2].provider == "anthropic"


def test_build_roster_skips_claude_if_unset(monkeypatch):
    monkeypatch.setenv("OLLAMA_TEACHER_MODELS", "gemma3:4b")
    monkeypatch.delenv("CLAUDE_TEACHER_MODEL", raising=False)
    roster = build_roster()
    assert all(t.provider == "ollama" for t in roster)


def test_build_roster_handles_empty_ollama_list(monkeypatch):
    monkeypatch.delenv("OLLAMA_TEACHER_MODELS", raising=False)
    monkeypatch.setenv("CLAUDE_TEACHER_MODEL", "claude-opus-5")
    roster = build_roster()
    assert len(roster) == 1
    assert roster[0].provider == "anthropic"


def test_teachers_for_category_filters():
    roster = [
        Teacher(name="a", provider="ollama", model="a", capabilities=["math", "reasoning"]),
        Teacher(name="b", provider="ollama", model="b", capabilities=["general"]),
        Teacher(name="c", provider="anthropic", model="c", capabilities=["math", "code"]),
    ]
    assert [t.name for t in teachers_for_category(roster, "math")] == ["a", "c"]
    assert [t.name for t in teachers_for_category(roster, "general")] == ["b"]
    assert teachers_for_category(roster, "nonexistent_category") == []


# --------------------------------------------------------------------------
# trajectory: to_dict + disagreement scoring
# --------------------------------------------------------------------------


def _make_trajectory(question, teacher, correct):
    return Trajectory(
        question=question, teacher=teacher, prompt_version="v1", answer="...",
        reasoning_type="arithmetic", tool_calls=[], verifier_result=correct,
        extracted_answer="4" if correct else "5", quality_score=1.0 if correct else 0.0,
    )


def test_trajectory_to_dict_shape():
    t = _make_trajectory("What is 2+2?", "ollama/gemma3:4b", True)
    d = t.to_dict()
    assert d["question"] == "What is 2+2?"
    assert d["teacher"] == "ollama/gemma3:4b"
    assert d["verifier_result"] is True
    assert d["disagreement_score"] is None  # not yet computed


def test_compute_disagreement_scores_mixed_correctness():
    trajectories = [
        _make_trajectory("Q1", "t1", True),
        _make_trajectory("Q1", "t2", True),
        _make_trajectory("Q1", "t3", False),
    ]
    compute_disagreement_scores(trajectories)
    for t in trajectories:
        assert t.disagreement_score == pytest.approx(1 / 3)


def test_compute_disagreement_scores_all_correct_is_zero():
    trajectories = [_make_trajectory("Q1", "t1", True), _make_trajectory("Q1", "t2", True)]
    compute_disagreement_scores(trajectories)
    assert all(t.disagreement_score == 0.0 for t in trajectories)


def test_compute_disagreement_scores_all_incorrect_is_one():
    trajectories = [_make_trajectory("Q1", "t1", False), _make_trajectory("Q1", "t2", False)]
    compute_disagreement_scores(trajectories)
    assert all(t.disagreement_score == 1.0 for t in trajectories)


def test_compute_disagreement_scores_independent_per_question():
    trajectories = [
        _make_trajectory("Q1", "t1", True),
        _make_trajectory("Q1", "t2", True),
        _make_trajectory("Q2", "t1", False),
        _make_trajectory("Q2", "t2", True),
    ]
    compute_disagreement_scores(trajectories)
    q1 = [t for t in trajectories if t.question == "Q1"]
    q2 = [t for t in trajectories if t.question == "Q2"]
    assert all(t.disagreement_score == 0.0 for t in q1)
    assert all(t.disagreement_score == pytest.approx(0.5) for t in q2)
