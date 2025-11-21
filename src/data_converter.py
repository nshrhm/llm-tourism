"""
データ変換ツール

このモジュールは、JSON形式の実験結果をCSV形式に変換します。
Excel互換（UTF-8 BOM付き）でのエクスポート機能を提供します。

Classes:
    DataConverter: JSON→CSV変換を管理

Example:
    >>> from src.data_converter import DataConverter
    >>> converter = DataConverter()
    >>> converter.convert_all_results()
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd
import yaml

# ロガーの設定
logger = logging.getLogger(__name__)


class DataConverter:
    """
    データ変換ツール

    JSON形式の実験結果をCSV形式に変換し、分析しやすい形式で保存します。

    Attributes:
        config (Dict[str, Any]): 設定情報
        raw_data_dir (Path): 生データディレクトリ
        processed_data_dir (Path): 処理済みデータディレクトリ
        output_filename (str): 出力CSVファイル名
        csv_encoding (str): CSV文字エンコーディング
    """

    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        """
        DataConverterの初期化

        Args:
            config_path: 設定ファイルのパス
        """
        # 設定ファイルの読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # データディレクトリの設定
        data_config = self.config.get("data", {})
        self.raw_data_dir = Path(data_config.get("raw_data_dir", "data/raw"))
        self.processed_data_dir = Path(
            data_config.get("processed_data_dir", "data/processed")
        )
        self.output_filename = data_config.get(
            "processed_file_name",
            "experiment_results.csv"
        )
        self.csv_encoding = data_config.get("csv_encoding", "utf-8-sig")

        # 出力ディレクトリの作成
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("DataConverterを初期化しました")

    def load_json_files(self) -> List[Dict[str, Any]]:
        """
        すべてのJSONファイルを読み込み

        Returns:
            実験結果データのリスト

        Raises:
            FileNotFoundError: データディレクトリが存在しない場合
        """
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"データディレクトリが見つかりません: {self.raw_data_dir}"
            )

        json_files = list(self.raw_data_dir.glob("*.json"))

        if not json_files:
            logger.warning("JSONファイルが見つかりません")
            return []

        logger.info(f"{len(json_files)}個のJSONファイルを発見しました")

        results = []
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                logger.warning(f"ファイル読み込みエラー [{json_file.name}]: {e}")

        logger.info(f"{len(results)}件のデータを読み込みました")
        return results

    def convert_to_dataframe(
        self,
        results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        JSON データをDataFrameに変換

        Args:
            results: 実験結果データのリスト

        Returns:
            pandas DataFrame
        """
        if not results:
            logger.warning("変換するデータがありません")
            return pd.DataFrame()

        # データ整形
        rows = []
        for result in results:
            row = {
                "session_id": result.get("session_id", ""),
                "experiment_id": result.get("experiment_id", ""),
                "timestamp": result.get("timestamp", ""),
                "model": result.get("model", {}).get("name", ""),
                "model_display_name": result.get("model", {}).get("display_name", ""),
                "persona_id": result.get("persona", {}).get("id", ""),
                "persona_name": result.get("persona", {}).get("name", ""),
                "travel_type_id": result.get("travel_type", {}).get("id", ""),
                "travel_type_name": result.get("travel_type", {}).get("name", ""),
                "prompt": result.get("prompt", ""),
                "response": result.get("response", ""),
                "tokens_used": result.get("performance", {}).get("tokens_used", 0),
                "latency_ms": result.get("performance", {}).get("latency_ms", 0),
            }
            rows.append(row)

        # DataFrameに変換
        df = pd.DataFrame(rows)

        # カラム順序の整理
        column_order = [
            "session_id",
            "experiment_id",
            "timestamp",
            "model",
            "model_display_name",
            "persona_id",
            "persona_name",
            "travel_type_id",
            "travel_type_name",
            "prompt",
            "response",
            "tokens_used",
            "latency_ms",
        ]

        df = df[column_order]

        logger.info(f"DataFrameに変換しました: {len(df)}行 × {len(df.columns)}列")
        return df

    def add_text_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        テキスト統計情報を追加

        Args:
            df: 元のDataFrame

        Returns:
            統計情報が追加されたDataFrame
        """
        if df.empty:
            return df

        logger.info("テキスト統計情報を計算中...")

        # レスポンスの文字数
        df["response_char_count"] = df["response"].str.len()

        # レスポンスの単語数（おおよその推定）
        # 日本語の場合、句読点で分割
        df["response_word_count"] = df["response"].apply(
            lambda x: len([s for s in str(x).replace("、", "。").split("。") if s.strip()])
        )

        # レスポンスの行数
        df["response_line_count"] = df["response"].str.count("\n") + 1

        # プロンプトの文字数
        df["prompt_char_count"] = df["prompt"].str.len()

        logger.info("テキスト統計情報を追加しました")
        return df

    def save_to_csv(
        self,
        df: pd.DataFrame,
        filename: str = None,
        add_timestamp: bool = False
    ) -> Path:
        """
        DataFrameをCSVファイルに保存

        Args:
            df: 保存するDataFrame
            filename: ファイル名（省略時は設定値を使用）
            add_timestamp: ファイル名にタイムスタンプを追加するか

        Returns:
            保存されたファイルのパス
        """
        if df.empty:
            logger.warning("保存するデータがありません")
            return None

        # ファイル名の決定
        if filename is None:
            filename = self.output_filename

        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = filename.rsplit(".", 1)
            filename = f"{name}_{timestamp}.{ext}"

        output_path = self.processed_data_dir / filename

        # CSV保存（UTF-8 BOM付き、Excel対応）
        df.to_csv(
            output_path,
            index=False,
            encoding=self.csv_encoding
        )

        logger.info(f"CSVファイルを保存しました: {output_path}")
        return output_path

    def convert_all_results(
        self,
        add_statistics: bool = True,
        add_timestamp: bool = False
    ) -> Dict[str, Any]:
        """
        すべての実験結果を変換してCSVに保存

        Args:
            add_statistics: テキスト統計情報を追加するか
            add_timestamp: ファイル名にタイムスタンプを追加するか

        Returns:
            変換結果の要約:
                - total_records: 総レコード数
                - output_file: 出力ファイルパス
                - columns: カラム数
                - timestamp: 変換実行時刻
        """
        logger.info("=" * 80)
        logger.info("データ変換を開始します")
        logger.info("=" * 80)

        # JSONファイル読み込み
        results = self.load_json_files()

        if not results:
            logger.error("変換するデータがありません")
            return {
                "total_records": 0,
                "output_file": None,
                "columns": 0,
                "timestamp": datetime.now().isoformat()
            }

        # DataFrame変換
        df = self.convert_to_dataframe(results)

        # テキスト統計追加
        if add_statistics:
            df = self.add_text_statistics(df)

        # CSV保存
        output_path = self.save_to_csv(df, add_timestamp=add_timestamp)

        # 要約情報
        summary = {
            "total_records": len(df),
            "output_file": str(output_path),
            "columns": len(df.columns),
            "timestamp": datetime.now().isoformat()
        }

        logger.info("=" * 80)
        logger.info("データ変換完了")
        logger.info(f"総レコード数: {summary['total_records']}")
        logger.info(f"カラム数: {summary['columns']}")
        logger.info(f"出力ファイル: {summary['output_file']}")
        logger.info("=" * 80)

        return summary

    def export_statistics_summary(self) -> pd.DataFrame:
        """
        統計サマリーをエクスポート

        Returns:
            統計サマリーのDataFrame
        """
        # データ読み込み
        results = self.load_json_files()
        df = self.convert_to_dataframe(results)

        if df.empty:
            return pd.DataFrame()

        df = self.add_text_statistics(df)

        # グループ別統計
        summary_list = []

        # モデル別統計
        model_stats = df.groupby("model").agg({
            "tokens_used": ["mean", "std", "min", "max"],
            "latency_ms": ["mean", "std", "min", "max"],
            "response_char_count": ["mean", "std", "min", "max"]
        }).round(2)

        # ペルソナ別統計
        persona_stats = df.groupby("persona_name").agg({
            "tokens_used": ["mean", "std"],
            "response_char_count": ["mean", "std"]
        }).round(2)

        # 旅行タイプ別統計
        travel_stats = df.groupby("travel_type_name").agg({
            "tokens_used": ["mean", "std"],
            "response_char_count": ["mean", "std"]
        }).round(2)

        # サマリー保存
        summary_path = self.processed_data_dir / "statistics_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("実験結果統計サマリー\n")
            f.write("=" * 80 + "\n\n")

            f.write("【モデル別統計】\n")
            f.write(model_stats.to_string())
            f.write("\n\n")

            f.write("【ペルソナ別統計】\n")
            f.write(persona_stats.to_string())
            f.write("\n\n")

            f.write("【旅行タイプ別統計】\n")
            f.write(travel_stats.to_string())
            f.write("\n")

        logger.info(f"統計サマリーを保存しました: {summary_path}")

        return df


if __name__ == "__main__":
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # データコンバーター初期化
        converter = DataConverter()

        print("\n" + "=" * 80)
        print("LLM地域観光紹介性能比較実験 - データ変換")
        print("=" * 80)

        # 全データ変換
        summary = converter.convert_all_results(
            add_statistics=True,
            add_timestamp=False
        )

        if summary["total_records"] > 0:
            print("\n" + "=" * 80)
            print("変換完了")
            print("=" * 80)
            print(f"総レコード数: {summary['total_records']}")
            print(f"カラム数: {summary['columns']}")
            print(f"出力ファイル: {summary['output_file']}")

            # 統計サマリーの出力
            print("\n統計サマリーを生成中...")
            converter.export_statistics_summary()
            print("✓ 統計サマリー生成完了")
        else:
            print("\n変換するデータがありません")
            print("先に実験を実行してください（src/experiment_runner.py）")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
