"""
全実験を自動実行（対話なし）

64パターンすべての実験を自動実行します。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment_runner import ExperimentRunner
from src.data_converter import DataConverter
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("  LLM地域観光紹介性能比較実験 - 全64パターン自動実行")
    print("=" * 80)
    print()

    try:
        # 実験ランナー初期化
        runner = ExperimentRunner()

        print("【ステップ1】実験実行")
        print("64パターンの実験を実行します...")
        print()

        # 全実験実行
        summary = runner.run_all_experiments()

        print()
        print("=" * 80)
        print("【実験完了】")
        print("=" * 80)
        print(f"成功: {summary['completed']}件")
        print(f"失敗: {summary['failed']}件")
        print(f"実行時間: {summary['duration_seconds']:.1f}秒")
        print()

        if summary['completed'] > 0:
            # データ変換
            print("【ステップ2】データ変換")
            converter = DataConverter()
            conv_summary = converter.convert_all_results(add_statistics=True)

            print(f"✓ データ変換完了")
            print(f"  - レコード数: {conv_summary['total_records']}")
            if conv_summary['output_file']:
                print(f"  - 出力ファイル: {conv_summary['output_file']}")
            print()

            # 英語出力チェック
            print("【ステップ3】英語出力チェック")
            import subprocess
            result = subprocess.run(
                ["python", "scripts/detect_english_output.py"],
                capture_output=True,
                text=True
            )
            print(result.stdout)

        print("=" * 80)
        print("  全処理完了")
        print("=" * 80)

        return 0 if summary['failed'] == 0 else 1

    except Exception as e:
        logger.error(f"エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
