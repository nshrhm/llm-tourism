"""
日本語出力テストスクリプト

修正したプロンプトで英語出力が発生していた2パターンをテストします。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_generator import PromptGenerator
from src.api_client import OpenRouterClient
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def detect_english_ratio(text: str) -> float:
    """テキスト中の英語の割合を推定"""
    if not text or not isinstance(text, str):
        return 0.0

    ascii_chars = len(re.findall(r'[a-zA-Z]', text))
    japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))
    total_chars = ascii_chars + japanese_chars

    if total_chars == 0:
        return 0.0

    return ascii_chars / total_chars


def main():
    print("=" * 80)
    print("  日本語出力テスト - 修正プロンプト検証")
    print("=" * 80)
    print()

    # テスト対象パターン
    test_patterns = [
        {
            "model": "google/gemini-2.5-flash",
            "model_display": "Google Gemini 2.5 Flash",
            "persona": "travel_agent",
            "persona_name": "旅行代理店スタッフ",
            "travel_type": "foreign_tourist",
            "travel_type_name": "外国人観光客（日本初訪問）"
        },
        {
            "model": "google/gemini-2.5-flash",
            "model_display": "Google Gemini 2.5 Flash",
            "persona": "government_official",
            "persona_name": "地方自治体の地域振興担当者",
            "travel_type": "foreign_tourist",
            "travel_type_name": "外国人観光客（日本初訪問）"
        }
    ]

    # プロンプト生成器とAPIクライアントの初期化
    print("【ステップ1】初期化")
    generator = PromptGenerator()
    client = OpenRouterClient()
    print("✓ 初期化完了")
    print()

    # 各パターンをテスト
    print("【ステップ2】テスト実行")
    print(f"テストパターン数: {len(test_patterns)}")
    print()

    results = []

    for i, pattern in enumerate(test_patterns, 1):
        print(f"テスト {i}/{len(test_patterns)}:")
        print(f"  モデル: {pattern['model_display']}")
        print(f"  ペルソナ: {pattern['persona_name']}")
        print(f"  旅行タイプ: {pattern['travel_type_name']}")

        # プロンプト生成
        all_prompts = generator.generate_all_prompts()
        target_prompt = None

        for p in all_prompts:
            if (p['model'] == pattern['model'] and
                p['persona_id'] == pattern['persona'] and
                p['travel_type_id'] == pattern['travel_type']):
                target_prompt = p
                break

        if not target_prompt:
            print("  ✗ プロンプトが見つかりません")
            continue

        print(f"  プロンプト冒頭: {target_prompt['prompt'][:80]}...")

        # API呼び出し
        try:
            response = client.generate(
                model=pattern['model'],
                prompt=target_prompt['prompt'],
                max_tokens=2000,
                temperature=0.7
            )

            # 成功時の処理
            text = response['response']
            char_count = len(text)
            english_ratio = detect_english_ratio(text)

            print(f"  ✓ 生成成功")
            print(f"  - 文字数: {char_count}")
            print(f"  - 英語割合: {english_ratio*100:.1f}%")
            print(f"  - レスポンス冒頭（100文字）:")
            print(f"    {text[:100]}...")
            print()

            results.append({
                'pattern': f"{pattern['persona_name']} × {pattern['travel_type_name']}",
                'english_ratio': english_ratio,
                'char_count': char_count,
                'success': True,
                'is_japanese': english_ratio < 0.3
            })

        except Exception as e:
            print(f"  ✗ エラー発生: {e}")
            results.append({
                'pattern': f"{pattern['persona_name']} × {pattern['travel_type_name']}",
                'success': False,
                'error': str(e)
            })

    # 結果サマリー
    print()
    print("=" * 80)
    print("【テスト結果サマリー】")
    print("=" * 80)
    print()

    success_count = sum(1 for r in results if r.get('success', False))
    japanese_count = sum(1 for r in results if r.get('is_japanese', False))

    print(f"テスト実行: {len(results)}/{len(test_patterns)}")
    print(f"成功: {success_count}/{len(results)}")
    print(f"日本語出力: {japanese_count}/{success_count}")
    print()

    if japanese_count == success_count and success_count == len(test_patterns):
        print("✓ すべてのパターンで日本語出力を確認しました！")
        print("✓ プロンプト修正が成功しています。")
        print()
        print("次のステップ: 全64パターンの再実験を実行してください。")
        print("  bash scripts/run_experiment.sh")
        return 0
    else:
        print("✗ 一部のパターンで問題が発生しました。")
        print("プロンプトの更なる調整が必要かもしれません。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
