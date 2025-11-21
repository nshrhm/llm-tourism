"""
OpenRouter APIクライアント

このモジュールは、OpenRouter API経由でLLMにアクセスするためのクライアントを提供します。
レート制限、リトライロジック、エラーハンドリングを含む堅牢な実装です。

Classes:
    OpenRouterClient: OpenRouter APIへのリクエストを管理するクライアント

Example:
    >>> from src.api_client import OpenRouterClient
    >>> client = OpenRouterClient()
    >>> response = client.generate(
    ...     model="openai/gpt-5.1-chat",
    ...     prompt="こんにちは"
    ... )
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import requests
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# ロガーの設定
logger = logging.getLogger(__name__)


class OpenRouterAPIError(Exception):
    """OpenRouter API関連のエラー"""
    pass


class OpenRouterClient:
    """
    OpenRouter APIクライアント

    OpenRouter経由で各種LLMにアクセスするためのクライアントクラス。
    レート制限、リトライ、エラーハンドリングを実装。

    Attributes:
        api_key (str): OpenRouter APIキー
        base_url (str): APIエンドポイントURL
        timeout (int): リクエストタイムアウト（秒）
        max_retries (int): 最大リトライ回数
        retry_delay (float): 初期リトライ待機時間（秒）
        backoff_factor (float): リトライ待機時間の増加率
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        backoff_factor: float = 2.0
    ):
        """
        OpenRouterClientの初期化

        Args:
            api_key: OpenRouter APIキー（省略時は環境変数から取得）
            base_url: APIエンドポイントURL
            timeout: リクエストタイムアウト（秒）
            max_retries: 最大リトライ回数
            retry_delay: 初期リトライ待機時間（秒）
            backoff_factor: リトライ待機時間の増加率（指数バックオフ）

        Raises:
            ValueError: APIキーが設定されていない場合
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter APIキーが設定されていません。"
                "環境変数OPENROUTER_API_KEYを設定するか、"
                "api_key引数で指定してください。"
            )

        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

        logger.info("OpenRouterClientを初期化しました")

    def _build_headers(self) -> Dict[str, str]:
        """
        HTTPリクエストヘッダーを構築

        Returns:
            HTTPヘッダーの辞書
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/llm-tourism-research",  # オプション
            "X-Title": "LLM Tourism Performance Comparison Study",  # オプション
        }

    def _build_payload(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> Dict[str, Any]:
        """
        APIリクエストのペイロードを構築

        Args:
            model: 使用するモデル名（例: "openai/gpt-5.1-chat"）
            prompt: 入力プロンプト
            temperature: 生成の創造性（0.0-1.0）
            max_tokens: 最大トークン数
            top_p: 核サンプリングパラメータ
            frequency_penalty: 頻度ペナルティ
            presence_penalty: 存在ペナルティ

        Returns:
            APIリクエスト用のペイロード辞書
        """
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        LLMにテキスト生成をリクエスト

        リトライロジックとエラーハンドリングを含む、堅牢なAPI呼び出し。

        Args:
            model: 使用するモデル名
            prompt: 入力プロンプト
            temperature: 生成の創造性（0.0-1.0）
            max_tokens: 最大トークン数
            **kwargs: その他のAPIパラメータ

        Returns:
            APIレスポンスを含む辞書:
                - response: 生成されたテキスト
                - model: 使用されたモデル名
                - tokens_used: 使用トークン数
                - latency_ms: 応答時間（ミリ秒）
                - timestamp: リクエスト時刻
                - raw_response: 生のAPIレスポンス

        Raises:
            OpenRouterAPIError: API呼び出しが失敗した場合
        """
        # ペイロード構築
        payload = self._build_payload(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        headers = self._build_headers()

        # リトライロジック
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                # リクエスト開始時刻
                start_time = time.time()
                timestamp = datetime.now().astimezone()

                logger.info(
                    f"APIリクエスト送信 (試行 {attempt + 1}/{self.max_retries}): "
                    f"model={model}"
                )

                # APIリクエスト
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                # 応答時間計算
                latency_ms = int((time.time() - start_time) * 1000)

                # HTTPエラーチェック
                response.raise_for_status()

                # レスポンスパース
                data = response.json()

                # エラーレスポンスチェック
                if "error" in data:
                    error_msg = data["error"].get("message", "不明なエラー")
                    raise OpenRouterAPIError(f"API Error: {error_msg}")

                # 成功時の処理
                generated_text = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)

                logger.info(
                    f"APIリクエスト成功: "
                    f"tokens={tokens_used}, latency={latency_ms}ms"
                )

                return {
                    "response": generated_text,
                    "model": model,
                    "tokens_used": tokens_used,
                    "latency_ms": latency_ms,
                    "timestamp": timestamp.isoformat(),
                    "raw_response": data
                }

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(
                    f"タイムアウト (試行 {attempt + 1}/{self.max_retries}): {e}"
                )

            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code

                # レート制限エラー（429）の場合は待機時間を延長
                if status_code == 429:
                    wait_time = self.retry_delay * (self.backoff_factor ** attempt)
                    logger.warning(
                        f"レート制限エラー (試行 {attempt + 1}/{self.max_retries}): "
                        f"{wait_time}秒待機します"
                    )
                    time.sleep(wait_time)
                    continue

                # その他のHTTPエラー
                logger.error(f"HTTPエラー {status_code}: {e}")

                # 4xxエラーはリトライしない
                if 400 <= status_code < 500:
                    raise OpenRouterAPIError(
                        f"クライアントエラー ({status_code}): {e}"
                    )

            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(
                    f"リクエストエラー (試行 {attempt + 1}/{self.max_retries}): {e}"
                )

            except (KeyError, ValueError) as e:
                last_exception = e
                logger.error(f"レスポンスパースエラー: {e}")
                raise OpenRouterAPIError(f"レスポンス解析失敗: {e}")

            # リトライ前の待機（最終試行を除く）
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (self.backoff_factor ** attempt)
                logger.info(f"{wait_time}秒待機してリトライします")
                time.sleep(wait_time)

        # すべてのリトライが失敗
        error_msg = f"APIリクエストが{self.max_retries}回失敗しました"
        logger.error(error_msg)
        raise OpenRouterAPIError(f"{error_msg}: {last_exception}")

    def test_connection(self) -> bool:
        """
        API接続テスト

        シンプルなプロンプトでAPIが正常に動作するか確認します。

        Returns:
            接続成功時True、失敗時False
        """
        try:
            logger.info("API接続テストを開始します")
            result = self.generate(
                model="openai/gpt-5.1-chat",
                prompt="こんにちは",
                max_tokens=50
            )
            logger.info("API接続テスト成功")
            return True
        except Exception as e:
            logger.error(f"API接続テスト失敗: {e}")
            return False


if __name__ == "__main__":
    # モジュール単体実行時のテスト
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        client = OpenRouterClient()
        print("OpenRouterClient初期化成功")

        # 接続テスト
        if client.test_connection():
            print("✓ API接続テスト成功")
        else:
            print("✗ API接続テスト失敗")

    except ValueError as e:
        print(f"エラー: {e}")
        print("環境変数OPENROUTER_API_KEYを設定してください")
