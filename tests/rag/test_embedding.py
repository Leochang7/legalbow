"""Unit tests for EmbeddingClient — mocked OpenAI API calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from legalbot.rag.embedding import EmbeddingClient


@pytest.fixture
def mock_openai():
    """Patch openai.AsyncOpenAI to return a mock client with dynamic responses."""
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        async def dynamic_embed(**kwargs):
            texts = kwargs.get("input", [])
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)])
                for i in range(len(texts))
            ]
            return mock_response

        mock_client.embeddings.create = dynamic_embed
        yield mock_client


class TestEmbeddingClient:

    async def test_embed_single_text(self, mock_openai):
        client = EmbeddingClient(model="text-embedding-3-small", api_key="test-key")
        result = await client.embed(["hello"])

        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]

    async def test_embed_batch(self, mock_openai):
        client = EmbeddingClient(model="text-embedding-3-small", api_key="test-key")
        result = await client.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.2, 0.4, 0.6]

    async def test_embed_empty_list(self, mock_openai):
        client = EmbeddingClient(model="text-embedding-3-small", api_key="test-key")
        result = await client.embed([])
        assert result == []

    async def test_dim_default(self):
        client = EmbeddingClient(model="text-embedding-3-small", api_key="test-key")
        assert client.dim() == 1536

    async def test_dim_dashscope(self):
        client = EmbeddingClient(model="text-embedding-v3", api_key="test-key")
        assert client.dim() == 1024

    async def test_dim_explicit(self):
        client = EmbeddingClient(model="custom-model", api_key="test-key", dim=768)
        assert client.dim() == 768

    async def test_api_base_passed(self):
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            async def mock_create(**kwargs):
                r = MagicMock()
                r.data = [MagicMock(embedding=[0.1])]
                return r

            mock_client.embeddings.create = mock_create

            client = EmbeddingClient(
                model="text-embedding-3-small",
                api_key="key",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            await client.embed(["test"])

            mock_cls.assert_called_once_with(
                api_key="key",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
