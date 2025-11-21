"""
実験実行エンジン

このモジュールは、全実験パターンを自動実行し、結果をJSON形式で保存します。
中断・再開機能、プログレス表示、エラーハンドリングを含みます。

Classes:
    ExperimentRunner: 実験の実行と結果保存を管理

Example:
    >>> from src.experiment_runner import ExperimentRunner
    >>> runner = ExperimentRunner()
    >>> runner.run_all_experiments()
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import yaml
from tqdm import tqdm

from src.api_client import OpenRouterClient
from src.prompt_generator import PromptGenerator

# ロガーの設定
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    実験実行エンジン

    プロンプト生成、API呼び出し、結果保存を統合的に管理します。

    Attributes:
        config (Dict[str, Any]): 実験設定
        api_client (OpenRouterClient): APIクライアント
        prompt_generator (PromptGenerator): プロンプト生成器
        output_dir (Path): 出力ディレクトリ
        checkpoint_file (Path): チェックポイントファイル
    """

    def __init__(
        self,
        config_path: str = "config/experiment_config.yaml",
        api_client: Optional[OpenRouterClient] = None
    ):
        """
        ExperimentRunnerの初期化

        Args:
            config_path: 設定ファイルのパス
            api_client: APIクライアント（省略時は新規作成）
        """
        # 設定ファイルの読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # コンポーネントの初期化
        self.prompt_generator = PromptGenerator(config_path)

        if api_client:
            self.api_client = api_client
        else:
            api_config = self.config.get("api", {})
            self.api_client = OpenRouterClient(
                base_url=api_config.get("base_url", "https://openrouter.ai/api/v1/chat/completions"),
                timeout=api_config.get("timeout", 60),
                max_retries=api_config.get("max_retries", 3),
                retry_delay=api_config.get("retry_delay", 5),
                backoff_factor=api_config.get("backoff_factor", 2.0)
            )

        # 出力ディレクトリの設定
        data_config = self.config.get("data", {})
        self.output_dir = Path(data_config.get("raw_data_dir", "data/raw"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # チェックポイントファイルの設定
        execution_config = self.config.get("execution", {})
        checkpoint_path = execution_config.get("checkpoint_file", "data/.checkpoint.json")
        self.checkpoint_file = Path(checkpoint_path)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # API設定の取得
        self.api_params = self.config.get("api", {}).get("parameters", {})
        self.request_delay = self.config.get("api", {}).get("request_delay", 2)

        logger.info("ExperimentRunnerを初期化しました")

    def _load_checkpoint(self) -> Dict[str, Any]:
        """
        チェックポイントの読み込み

        Returns:
            チェックポイントデータ（存在しない場合は空辞書）
        """
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                logger.info(
                    f"チェックポイントを読み込みました: "
                    f"{checkpoint.get('completed_count', 0)}件完了"
                )
                return checkpoint
            except Exception as e:
                logger.warning(f"チェックポイント読み込みエラー: {e}")

        return {"completed_experiments": [], "completed_count": 0}

    def _save_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        チェックポイントの保存

        Args:
            checkpoint: チェックポイントデータ
        """
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            logger.debug("チェックポイントを保存しました")
        except Exception as e:
            logger.warning(f"チェックポイント保存エラー: {e}")

    def _save_result(
        self,
        experiment_info: Dict[str, Any],
        api_response: Dict[str, Any],
        session_id: str
    ):
        """
        実験結果をJSONファイルに保存

        Args:
            experiment_info: 実験情報
            api_response: APIレスポンス
            session_id: セッションID
        """
        # 結果データの構築
        result = {
            "session_id": session_id,
            "experiment_id": experiment_info["experiment_id"],
            "timestamp": api_response["timestamp"],
            "model": {
                "name": experiment_info["model"],
                "display_name": experiment_info["model_display_name"]
            },
            "persona": {
                "id": experiment_info["persona_id"],
                "name": experiment_info["persona_name"]
            },
            "travel_type": {
                "id": experiment_info["travel_type_id"],
                "name": experiment_info["travel_type_name"]
            },
            "prompt": experiment_info["prompt"],
            "response": api_response["response"],
            "performance": {
                "tokens_used": api_response["tokens_used"],
                "latency_ms": api_response["latency_ms"]
            },
            "metadata": experiment_info["metadata"],
            "api_parameters": self.api_params
        }

        # ファイル名生成
        exp_id = experiment_info["experiment_id"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{exp_id}_{timestamp}.json"
        filepath = self.output_dir / filename

        # ファイル保存
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.debug(f"結果を保存しました: {filename}")

    def run_single_experiment(
        self,
        experiment_info: Dict[str, Any],
        session_id: str
    ) -> bool:
        """
        単一の実験を実行

        Args:
            experiment_info: 実験情報
            session_id: セッションID

        Returns:
            成功時True、失敗時False
        """
        try:
            # API呼び出し
            response = self.api_client.generate(
                model=experiment_info["model"],
                prompt=experiment_info["prompt"],
                **self.api_params
            )

            # 結果保存
            self._save_result(experiment_info, response, session_id)

            return True

        except Exception as e:
            logger.error(
                f"実験失敗 [{experiment_info['experiment_id']}]: {e}"
            )
            return False

    def run_all_experiments(
        self,
        resume: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        全実験パターンを実行

        Args:
            resume: チェックポイントから再開するか
            show_progress: プログレスバーを表示するか

        Returns:
            実行結果の要約:
                - session_id: セッションID
                - total_experiments: 総実験数
                - completed: 完了数
                - failed: 失敗数
                - duration_seconds: 実行時間（秒）
                - start_time: 開始時刻
                - end_time: 終了時刻
        """
        # セッションID生成
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info("=" * 80)
        logger.info(f"実験セッション開始: {session_id}")
        logger.info("=" * 80)

        # 全プロンプト生成
        all_prompts = self.prompt_generator.generate_all_prompts()
        total_count = len(all_prompts)

        # チェックポイント読み込み
        checkpoint = {"completed_experiments": [], "completed_count": 0}
        if resume:
            checkpoint = self._load_checkpoint()

        # 未完了の実験をフィルタ
        completed_ids = set(checkpoint.get("completed_experiments", []))
        pending_prompts = [
            p for p in all_prompts
            if p["experiment_id"] not in completed_ids
        ]

        logger.info(
            f"実験数: {total_count}件 "
            f"(完了: {len(completed_ids)}件, 残り: {len(pending_prompts)}件)"
        )

        # 実行開始
        start_time = datetime.now()
        success_count = len(completed_ids)
        failure_count = 0

        # プログレスバー設定
        if show_progress:
            progress_bar = tqdm(
                total=total_count,
                initial=len(completed_ids),
                desc="実験実行中",
                unit="実験"
            )
        else:
            progress_bar = None

        # 実験実行ループ
        for i, prompt_info in enumerate(pending_prompts):
            exp_id = prompt_info["experiment_id"]

            logger.info(
                f"[{success_count + failure_count + 1}/{total_count}] "
                f"実験実行: {exp_id}"
            )

            # 実験実行
            success = self.run_single_experiment(prompt_info, session_id)

            if success:
                success_count += 1
                # チェックポイント更新
                checkpoint["completed_experiments"].append(exp_id)
                checkpoint["completed_count"] = success_count
                self._save_checkpoint(checkpoint)
            else:
                failure_count += 1

            # プログレスバー更新
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "成功": success_count,
                    "失敗": failure_count
                })

            # レート制限対策（最後の実験以外）
            if i < len(pending_prompts) - 1:
                time.sleep(self.request_delay)

        # プログレスバー終了
        if progress_bar:
            progress_bar.close()

        # 実行終了
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # チェックポイント削除（全完了時）
        if success_count == total_count:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.info("チェックポイントを削除しました（全実験完了）")

        # 結果要約
        summary = {
            "session_id": session_id,
            "total_experiments": total_count,
            "completed": success_count,
            "failed": failure_count,
            "duration_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }

        logger.info("=" * 80)
        logger.info("実験セッション完了")
        logger.info(f"総実験数: {total_count}件")
        logger.info(f"成功: {success_count}件")
        logger.info(f"失敗: {failure_count}件")
        logger.info(f"実行時間: {duration:.1f}秒")
        logger.info("=" * 80)

        return summary

    def run_filtered_experiments(
        self,
        model: str = None,
        persona_id: str = None,
        travel_type_id: str = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        フィルタリングされた実験を実行

        Args:
            model: 特定のモデルのみ実行
            persona_id: 特定のペルソナのみ実行
            travel_type_id: 特定の旅行タイプのみ実行
            show_progress: プログレスバーを表示するか

        Returns:
            実行結果の要約
        """
        # セッションID生成
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_filtered"

        # プロンプト生成とフィルタリング
        all_prompts = self.prompt_generator.generate_all_prompts()
        filtered_prompts = self.prompt_generator.filter_prompts(
            all_prompts,
            model=model,
            persona_id=persona_id,
            travel_type_id=travel_type_id
        )

        logger.info(
            f"フィルタリング実験開始: {len(filtered_prompts)}件"
        )

        # 実行
        start_time = datetime.now()
        success_count = 0
        failure_count = 0

        if show_progress:
            progress_bar = tqdm(
                total=len(filtered_prompts),
                desc="実験実行中",
                unit="実験"
            )
        else:
            progress_bar = None

        for i, prompt_info in enumerate(filtered_prompts):
            exp_id = prompt_info["experiment_id"]

            success = self.run_single_experiment(prompt_info, session_id)

            if success:
                success_count += 1
            else:
                failure_count += 1

            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "成功": success_count,
                    "失敗": failure_count
                })

            # レート制限対策
            if i < len(filtered_prompts) - 1:
                time.sleep(self.request_delay)

        if progress_bar:
            progress_bar.close()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        summary = {
            "session_id": session_id,
            "total_experiments": len(filtered_prompts),
            "completed": success_count,
            "failed": failure_count,
            "duration_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }

        logger.info(f"フィルタリング実験完了: 成功={success_count}, 失敗={failure_count}")

        return summary


if __name__ == "__main__":
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # 実験ランナー初期化
        runner = ExperimentRunner()

        print("\n" + "=" * 80)
        print("LLM地域観光紹介性能比較実験")
        print("=" * 80)

        # ユーザー確認
        print("\n実験を開始しますか？")
        print("- 64パターンの実験を実行します")
        print("- APIコールが発生します（費用が発生する可能性があります）")
        response = input("実行しますか？ (yes/no): ")

        if response.lower() in ["yes", "y"]:
            # 全実験実行
            summary = runner.run_all_experiments()

            print("\n" + "=" * 80)
            print("実験完了")
            print("=" * 80)
            print(f"成功: {summary['completed']}件")
            print(f"失敗: {summary['failed']}件")
            print(f"実行時間: {summary['duration_seconds']:.1f}秒")
        else:
            print("実験をキャンセルしました")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
