"""
APIクライアントの単体テスト

OpenRouterClientの各機能をテストします。

実行方法:
    pytest tests/test_api_client.py
    pytest tests/test_api_client.py -v  # 詳細表示
    pytest tests/test_api_client.py -k test_init  # 特定のテストのみ実行
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from src.api_client import OpenRouterClient, OpenRouterAPIError


class TestOpenRouterClientInit:
    """OpenRouterClientの初期化テスト"""

    def test_init_with_api_key(self):
        """APIキーを指定して初期化"""
        client = OpenRouterClient(api_key="test_key_123")
        assert client.api_key == "test_key_123"
        assert client.base_url == "https://openrouter.ai/api/v1/chat/completions"
        assert client.timeout == 60
        assert client.max_retries == 3

    def test_init_with_env_variable(self, monkeypatch):
        """環境変数からAPIキーを取得"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env_key_456")
        client = OpenRouterClient()
        assert client.api_key == "env_key_456"

    def test_init_without_api_key(self, monkeypatch):
        """APIキーなしで初期化（エラー）"""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="APIキーが設定されていません"):
            OpenRouterClient(api_key=None)

    def test_init_with_custom_parameters(self):
        """カスタムパラメータで初期化"""
        client = OpenRouterClient(
            api_key="test_key",
            timeout=120,
            max_retries=5,
            retry_delay=10.0,
            backoff_factor=3.0
        )
        assert client.timeout == 120
        assert client.max_retries == 5
        assert client.retry_delay == 10.0
        assert client.backoff_factor == 3.0


class TestOpenRouterClientHelpers:
    """ヘルパーメソッドのテスト"""

    def test_build_headers(self):
        """HTTPヘッダーの構築"""
        client = OpenRouterClient(api_key="test_key")
        headers = client._build_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_key"
        assert headers["Content-Type"] == "application/json"

    def test_build_payload(self):
        """APIペイロードの構築"""
        client = OpenRouterClient(api_key="test_key")
        payload = client._build_payload(
            model="openai/gpt-5.1-chat",
            prompt="テストプロンプト",
            temperature=0.8,
            max_tokens=1000
        )

        assert payload["model"] == "openai/gpt-5.1-chat"
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "テストプロンプト"
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 1000


class TestOpenRouterClientGenerate:
    """generate メソッドのテスト"""

    @patch('src.api_client.requests.post')
    def test_generate_success(self, mock_post):
        """正常なAPI呼び出し"""
        # モックレスポンスの設定
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "下関は素晴らしい観光地です。"
                    }
                }
            ],
            "usage": {
                "total_tokens": 150
            }
        }
        mock_post.return_value = mock_response

        # テスト実行
        client = OpenRouterClient(api_key="test_key")
        result = client.generate(
            model="openai/gpt-5.1-chat",
            prompt="下関を紹介してください"
        )

        # 検証
        assert result["response"] == "下関は素晴らしい観光地です。"
        assert result["model"] == "openai/gpt-5.1-chat"
        assert result["tokens_used"] == 150
        assert "latency_ms" in result
        assert "timestamp" in result

    @patch('src.api_client.requests.post')
    def test_generate_timeout(self, mock_post):
        """タイムアウトエラー"""
        mock_post.side_effect = requests.exceptions.Timeout("Timeout")

        client = OpenRouterClient(api_key="test_key", max_retries=2)

        with pytest.raises(OpenRouterAPIError, match="APIリクエストが2回失敗"):
            client.generate(
                model="openai/gpt-5.1-chat",
                prompt="テスト"
            )

    @patch('src.api_client.requests.post')
    def test_generate_http_error_4xx(self, mock_post):
        """4xxエラー（クライアントエラー）"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.HTTPError(response=mock_response)
        )
        mock_post.return_value = mock_response

        client = OpenRouterClient(api_key="test_key")

        with pytest.raises(OpenRouterAPIError, match="クライアントエラー"):
            client.generate(
                model="invalid-model",
                prompt="テスト"
            )

    @patch('src.api_client.requests.post')
    @patch('src.api_client.time.sleep')  # sleepをモック化してテスト高速化
    def test_generate_rate_limit_retry(self, mock_sleep, mock_post):
        """レート制限エラーのリトライ"""
        # 1回目: 429エラー、2回目: 成功
        mock_error_response = Mock()
        mock_error_response.status_code = 429
        mock_error_response.raise_for_status.side_effect = (
            requests.exceptions.HTTPError(response=mock_error_response)
        )

        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "choices": [{"message": {"content": "成功"}}],
            "usage": {"total_tokens": 50}
        }

        mock_post.side_effect = [mock_error_response, mock_success_response]

        client = OpenRouterClient(api_key="test_key", max_retries=3)
        result = client.generate(model="openai/gpt-5.1-chat", prompt="テスト")

        assert result["response"] == "成功"
        assert mock_sleep.called  # sleepが呼ばれたことを確認

    @patch('src.api_client.requests.post')
    def test_generate_api_error_response(self, mock_post):
        """APIエラーレスポンス"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid model specified"
            }
        }
        mock_post.return_value = mock_response

        client = OpenRouterClient(api_key="test_key")

        with pytest.raises(OpenRouterAPIError, match="API Error"):
            client.generate(
                model="invalid-model",
                prompt="テスト"
            )


class TestOpenRouterClientIntegration:
    """統合テスト（実際のAPI呼び出しを含む）"""

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEYが設定されていません"
    )
    @pytest.mark.integration
    def test_real_api_call(self):
        """実際のAPI呼び出しテスト（スキップ可能）"""
        client = OpenRouterClient()

        result = client.generate(
            model="openai/gpt-5.1-chat",
            prompt="「こんにちは」と返答してください",
            max_tokens=50
        )

        assert "response" in result
        assert len(result["response"]) > 0
        assert result["tokens_used"] > 0

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEYが設定されていません"
    )
    @pytest.mark.integration
    def test_connection_test(self):
        """接続テスト"""
        client = OpenRouterClient()
        assert client.test_connection() is True


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"])
