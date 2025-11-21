"""
英語出力パターンの検出スクリプト

実験データから英語出力になっているパターンを特定します。
"""

import pandas as pd
import re
from pathlib import Path


def detect_english_ratio(text: str) -> float:
    """
    テキスト中の英語の割合を推定

    Args:
        text: 分析対象テキスト

    Returns:
        英語の割合（0.0-1.0）
    """
    if not text or not isinstance(text, str):
        return 0.0

    # アルファベットの文字数をカウント
    ascii_chars = len(re.findall(r'[a-zA-Z]', text))

    # 日本語文字（ひらがな、カタカナ、漢字）をカウント
    japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))

    # 総文字数（空白を除く）
    total_chars = ascii_chars + japanese_chars

    if total_chars == 0:
        return 0.0

    # 英語の割合を計算
    english_ratio = ascii_chars / total_chars

    return english_ratio


def main():
    # データ読み込み
    csv_path = Path('data/processed/experiment_results.csv')

    if not csv_path.exists():
        print(f"エラー: {csv_path} が見つかりません")
        return

    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("  英語出力パターンの検出")
    print("=" * 80)
    print()

    # 各レコードの英語割合を計算
    df['english_ratio'] = df['response'].apply(detect_english_ratio)

    # 英語が30%以上含まれるものを抽出（閾値は調整可能）
    threshold = 0.3
    english_outputs = df[df['english_ratio'] >= threshold].copy()

    print(f"検出閾値: 英語割合 {threshold*100:.0f}% 以上")
    print(f"該当パターン数: {len(english_outputs)}/{len(df)}")
    print()

    if len(english_outputs) > 0:
        print("【英語出力が検出されたパターン】")
        print()

        for idx, row in english_outputs.iterrows():
            print(f"パターン {idx + 1}:")
            print(f"  - モデル: {row['model_display_name']}")
            print(f"  - ペルソナ: {row['persona_name']}")
            print(f"  - 旅行タイプ: {row['travel_type_name']}")
            print(f"  - 英語割合: {row['english_ratio']*100:.1f}%")
            print(f"  - 文字数: {row['response_char_count']}")
            print(f"  - レスポンス冒頭（100文字）:")
            print(f"    {row['response'][:100]}...")
            print()

        # 組み合わせのサマリー
        print("【パターンサマリー】")
        print()

        # モデル別
        model_counts = english_outputs['model_display_name'].value_counts()
        print("モデル別:")
        for model, count in model_counts.items():
            print(f"  - {model}: {count}件")
        print()

        # ペルソナ別
        persona_counts = english_outputs['persona_name'].value_counts()
        print("ペルソナ別:")
        for persona, count in persona_counts.items():
            print(f"  - {persona}: {count}件")
        print()

        # 旅行タイプ別
        travel_counts = english_outputs['travel_type_name'].value_counts()
        print("旅行タイプ別:")
        for travel, count in travel_counts.items():
            print(f"  - {travel}: {count}件")
        print()

        # CSV出力
        output_path = Path('data/processed/english_output_patterns.csv')
        english_outputs.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"詳細データを保存: {output_path}")

    else:
        print("✓ 英語出力は検出されませんでした")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
