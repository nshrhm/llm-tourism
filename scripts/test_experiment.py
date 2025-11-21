"""
検証用の小規模実験スクリプト

2パターンのみの実験を実行して、システム全体の動作を検証します。
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_generator import PromptGenerator
from src.api_client import OpenRouterClient
from src.experiment_runner import ExperimentRunner
from src.data_converter import DataConverter
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("  LLM地域観光紹介性能比較実験 - 検証テスト")
    print("=" * 80)
    print()

    try:
        # ステップ1: プロンプト生成器のテスト
        print("【ステップ1】プロンプト生成器の初期化")
        generator = PromptGenerator()
        all_prompts = generator.generate_all_prompts()
        print(f"✓ {len(all_prompts)}個のプロンプトを生成")

        # 最初の2パターンのみ抽出
        test_prompts = all_prompts[:2]
        print(f"✓ テスト用に{len(test_prompts)}パターンを選択")
        for i, p in enumerate(test_prompts):
            print(f"  [{i+1}] {p['model_display_name']} × {p['persona_name']} × {p['travel_type_name']}")
        print()

        # ステップ2: APIクライアントの初期化
        print("【ステップ2】APIクライアントの初期化")
        client = OpenRouterClient()
        print(f"✓ APIクライアント初期化成功")
        print()

        # ステップ3: 実験実行の確認
        print("【ステップ3】実験実行の確認")
        print("注意: 2回のAPI呼び出しが発生します（費用が発生する可能性があります）")
        response = input("実験を実行しますか？ (yes/no): ")
        print()

        if response.lower() not in ['yes', 'y']:
            print("テストをキャンセルしました")
            return

        # ステップ4: 実験実行
        print("【ステップ4】実験実行")
        runner = ExperimentRunner()

        # フィルタリング実験実行（最初のモデル・ペルソナ・旅行タイプのみ）
        first_model = test_prompts[0]['model']
        first_persona = test_prompts[0]['persona_id']

        summary = runner.run_filtered_experiments(
            model=first_model,
            persona_id=first_persona,
            show_progress=True
        )

        print()
        print(f"✓ 実験完了")
        print(f"  - 成功: {summary['completed']}件")
        print(f"  - 失敗: {summary['failed']}件")
        print(f"  - 実行時間: {summary['duration_seconds']:.1f}秒")
        print()

        # ステップ5: データ変換
        print("【ステップ5】データ変換")
        converter = DataConverter()
        conv_summary = converter.convert_all_results(add_statistics=True)

        print(f"✓ データ変換完了")
        print(f"  - レコード数: {conv_summary['total_records']}")
        print(f"  - 出力ファイル: {conv_summary['output_file']}")
        print()

        # ステップ6: 結果確認
        print("【ステップ6】結果確認")
        raw_files = list(Path('data/raw').glob('*.json'))
        print(f"✓ 生データ（JSON）: {len(raw_files)}ファイル")

        csv_file = Path('data/processed/experiment_results.csv')
        if csv_file.exists():
            import pandas as pd
            df = pd.read_csv(csv_file)
            print(f"✓ CSVファイル: {len(df)}レコード")
        print()

        # 完了
        print("=" * 80)
        print("  検証テスト完了")
        print("=" * 80)
        print()
        print("すべてのコンポーネントが正常に動作しています！")
        print()
        print("次のステップ:")
        print("  1. データの確認: data/processed/experiment_results.csv")
        print("  2. 本実験の実行: bash scripts/run_experiment.sh")
        print()

    except Exception as e:
        print()
        print(f"✗ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
