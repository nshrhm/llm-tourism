"""
自動検証スクリプト

ユーザー入力なしで小規模実験を自動実行します。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_generator import PromptGenerator
from src.api_client import OpenRouterClient
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
    print("  LLM地域観光紹介性能比較実験 - 自動検証テスト")
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

        # ステップ3: 実験実行（自動）
        print("【ステップ3】実験実行（自動モード）")
        print(f"実験パターン数: {len(test_prompts)}")
        print()

        runner = ExperimentRunner()

        # フィルタリング実験実行（最初のモデル・ペルソナ・旅行タイプのみ）
        first_model = test_prompts[0]['model']
        first_persona = test_prompts[0]['persona_id']

        print(f"実験開始: モデル={first_model}, ペルソナ={first_persona}")
        print()

        summary = runner.run_filtered_experiments(
            model=first_model,
            persona_id=first_persona,
            show_progress=True
        )

        print()
        print("【実験結果】")
        print(f"  - 成功: {summary['completed']}件")
        print(f"  - 失敗: {summary['failed']}件")
        print(f"  - 実行時間: {summary['duration_seconds']:.1f}秒")
        print()

        if summary['completed'] == 0:
            print("✗ 実験が1つも成功しませんでした")
            print("エラーログを確認してください")
            return 1

        # ステップ4: データ変換
        print("【ステップ4】データ変換")
        converter = DataConverter()
        conv_summary = converter.convert_all_results(add_statistics=True)

        print(f"✓ データ変換完了")
        print(f"  - レコード数: {conv_summary['total_records']}")
        if conv_summary['output_file']:
            print(f"  - 出力ファイル: {conv_summary['output_file']}")
        print()

        # ステップ5: 結果確認
        print("【ステップ5】結果確認")
        raw_files = list(Path('data/raw').glob('*.json'))
        print(f"✓ 生データ（JSON）: {len(raw_files)}ファイル")

        csv_file = Path('data/processed/experiment_results.csv')
        if csv_file.exists():
            import pandas as pd
            df = pd.read_csv(csv_file)
            print(f"✓ CSVファイル: {len(df)}レコード")
            print()

            # サンプルデータ表示
            print("【サンプルデータ】")
            sample = df.iloc[0]
            print(f"モデル: {sample['model_display_name']}")
            print(f"ペルソナ: {sample['persona_name']}")
            print(f"旅行タイプ: {sample['travel_type_name']}")
            print(f"文字数: {sample['response_char_count']}")
            print(f"トークン数: {sample['tokens_used']}")
            print(f"応答時間: {sample['latency_ms']}ms")
            print()
            print(f"レスポンス（最初の200文字）:")
            print(sample['response'][:200] + "...")
        print()

        # 完了
        print("=" * 80)
        print("  自動検証テスト完了")
        print("=" * 80)
        print()
        print("✓ すべてのコンポーネントが正常に動作しています！")
        print()
        print("次のステップ:")
        print("  1. データの確認: data/processed/experiment_results.csv")
        print("  2. 本実験の実行: bash scripts/run_experiment.sh")
        print()

        return 0

    except Exception as e:
        print()
        print(f"✗ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
