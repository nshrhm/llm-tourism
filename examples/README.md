# サンプル実験結果

このディレクトリには、実験システムの出力例が含まれています。

## ファイル一覧

### 1. sample_response.json

**概要**: 単一の実験結果（生データ）のサンプル

**形式**: JSON

**内容**:
- **実験ID**: `exp_000`
- **モデル**: GPT-5.1
- **ペルソナ**: 家族連れ（小学生の子供2人）
- **旅行タイプ**: 観光・名所巡り
- **トークン数**: 710トークン
- **応答時間**: 6,542ms（約6.5秒）

**用途**:
- 実験システムの生データフォーマットの確認
- API応答の構造理解
- データ処理プログラムの開発・テスト

**サンプル構造**:
```json
{
  "experiment_id": "exp_000",
  "timestamp": "2025-01-21T10:15:32.456789",
  "model": "openai/gpt-5.1",
  "model_display_name": "GPT-5.1",
  "persona_id": "persona_family",
  "persona_name": "家族連れ（小学生の子供2人）",
  "travel_type_id": "type_sightseeing",
  "travel_type_name": "観光・名所巡り",
  "prompt": "...",
  "response": "...",
  "api_response": { ... },
  "tokens_used": 710,
  "latency_ms": 6542,
  "success": true,
  "error": null
}
```

### 2. sample_results.csv

**概要**: 複数の実験結果（処理済みデータ）のサンプル

**形式**: CSV（UTF-8 BOM）

**内容**: 4つの実験パターン
1. GPT-5.1 × 家族連れ × 観光・名所巡り
2. Claude Sonnet 4.5 × 家族連れ × 観光・名所巡り
3. Gemini 2.5 Flash × シニア夫婦 × グルメ・食べ歩き
4. Grok-4 Fast × 一人旅 × 歴史・文化体験

**統計情報**:
| モデル | トークン数 | 文字数 | 応答時間 |
|--------|-----------|--------|---------|
| GPT-5.1 | 710 | 635 | 6,542ms |
| Claude Sonnet 4.5 | 592 | 672 | 5,821ms |
| Gemini 2.5 Flash | 654 | 708 | 4,987ms |
| Grok-4 Fast | 718 | 759 | 7,234ms |

**用途**:
- 実験結果CSVのフォーマット確認
- データ分析コードの開発・テスト
- Excel/pandas での分析練習

**CSV列構成**:
```
experiment_id, timestamp, model, model_display_name,
persona_id, persona_name, travel_type_id, travel_type_name,
prompt, response, tokens_used, latency_ms, success, error,
response_char_count, response_word_count, response_line_count
```

## 使用方法

### Pythonでの読み込み例

#### JSON ファイル
```python
import json
from pathlib import Path

# JSONファイルの読み込み
with open('examples/sample_response.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"モデル: {data['model_display_name']}")
print(f"ペルソナ: {data['persona_name']}")
print(f"トークン数: {data['tokens_used']}")
print(f"応答: {data['response'][:100]}...")
```

#### CSV ファイル
```python
import pandas as pd

# CSVファイルの読み込み
df = pd.read_csv('examples/sample_results.csv')

# 基本統計
print(df[['model_display_name', 'tokens_used', 'latency_ms']].describe())

# モデル別平均トークン数
print(df.groupby('model_display_name')['tokens_used'].mean())

# ペルソナ別文字数分布
print(df.groupby('persona_name')['response_char_count'].describe())
```

### 分析例

#### 1. モデル間のパフォーマンス比較
```python
import matplotlib.pyplot as plt

# 応答時間の比較
df.plot(x='model_display_name', y='latency_ms', kind='bar',
        title='モデル別応答時間')
plt.ylabel('応答時間 (ms)')
plt.tight_layout()
plt.savefig('latency_comparison.png')
```

#### 2. トークン数と文字数の関係
```python
# 散布図作成
plt.scatter(df['tokens_used'], df['response_char_count'])
plt.xlabel('トークン数')
plt.ylabel('文字数')
plt.title('トークン数と文字数の関係')
plt.savefig('tokens_vs_chars.png')
```

#### 3. ペルソナ別応答の長さ
```python
# ペルソナごとの平均文字数
persona_stats = df.groupby('persona_name').agg({
    'response_char_count': 'mean',
    'tokens_used': 'mean',
    'latency_ms': 'mean'
})
print(persona_stats)
```

## 注意事項

### データの性質

1. **サンプルデータ**: これらは実験システムのフォーマット例であり、実際のAPI応答を簡略化・編集している場合があります。

2. **タイムスタンプ**: サンプル内のタイムスタンプは例示目的です。実際の実験では現在時刻が記録されます。

3. **文字数・トークン数**: 言語モデルによってトークナイザーが異なるため、同じ文字数でもトークン数は変動します。

4. **応答内容**: サンプルの応答内容は下関観光に関する一般的な情報です。実際の実験では、モデルやペルソナによって異なる応答が生成されます。

### 本実験との違い

- **サンプル**: 4パターンのみ
- **本実験**: 64パターン（4モデル × 4ペルソナ × 4旅行タイプ）

## ライセンス

このサンプルデータは、プロジェクト本体と同じ [CC BY 4.0](../LICENSE) ライセンスの下で提供されます。

## 関連ドキュメント

- [README.md](../README.md) - プロジェクト概要
- [docs/manual.md](../docs/manual.md) - データ分析マニュアル
- [docs/methodology.md](../docs/methodology.md) - 実験方法論
- [CONTRIBUTING.md](../CONTRIBUTING.md) - 貢献ガイドライン
