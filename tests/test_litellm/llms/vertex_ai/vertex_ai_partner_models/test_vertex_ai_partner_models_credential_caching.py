import sys
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
    VertexAIPartnerModels,
)


@pytest.fixture(autouse=True)
def mock_vertexai():
    """Stub out vertexai so completion() passes the import guard."""
    mock = MagicMock()
    mock.preview = MagicMock()
    with patch.dict(sys.modules, {"vertexai": mock}):
        yield mock


def _make_handler_with_mock_vertex_llm():
    handler = VertexAIPartnerModels()
    mock_vertex_llm = MagicMock()
    mock_vertex_llm._ensure_access_token.return_value = ("fake-token", "fake-project")
    handler._cached_vertex_llm = mock_vertex_llm
    return handler, mock_vertex_llm


def _common_completion_kwargs():
    return dict(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        model_response=MagicMock(),
        print_verbose=MagicMock(),
        encoding=MagicMock(),
        logging_obj=MagicMock(),
        api_base=None,
        optional_params={},
        custom_prompt_dict={},
        headers=None,
        timeout=10,
        litellm_params={},
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials="{}",
    )


@patch("litellm.llms.anthropic.chat.AnthropicChatCompletion")
def test_vertex_llm_instance_reused_across_completion_calls(mock_anthropic):
    """
    The cached VertexLLM instance should be reused across multiple completion()
    calls so that _credentials_project_mapping persists between requests.
    """
    handler, mock_vertex_llm = _make_handler_with_mock_vertex_llm()

    handler.completion(**_common_completion_kwargs())
    handler.completion(**_common_completion_kwargs())

    assert handler._cached_vertex_llm is mock_vertex_llm
    assert mock_vertex_llm._ensure_access_token.call_count == 2


def test_cached_vertex_llm_starts_as_none():
    handler = VertexAIPartnerModels()
    assert handler._cached_vertex_llm is None
