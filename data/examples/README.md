# サンプルデータ

このディレクトリには、実験結果のサンプルデータが含まれています。

## ファイル一覧

### JSON形式（生データ）

各LLMモデルから1パターンずつ、計4件のサンプルを提供しています：

- **exp_000_gpt5_sample.json** - OpenAI GPT-5.1
- **exp_016_claude_sample.json** - Anthropic Claude Sonnet 4.5
- **exp_032_gemini_sample.json** - Google Gemini 2.5 Flash
- **exp_048_grok_sample.json** - xAI Grok-4 Fast

すべて以下の条件で生成されています：
- **ペルソナ**: 旅行代理店スタッフ
- **旅行タイプ**: 家族旅行

### CSV形式（処理済みデータ）

- **sample_results.csv** - 上記4パターンの処理済みデータ

## データ構造

### JSON形式

```json
{
  "session_id": "セッションID",
  "experiment_id": "実験ID（exp_000-063）",
  "timestamp": "ISO 8601形式のタイムスタンプ",
  "model": {
    "name": "モデル名",
    "display_name": "表示名"
  },
  "persona": {
    "id": "ペルソナID",
    "name": "ペルソナ名"
  },
  "travel_type": {
    "id": "旅行タイプID",
    "name": "旅行タイプ名"
  },
  "prompt": "LLMに送信したプロンプト",
  "response": "LLMからの応答テキスト",
  "performance": {
    "tokens_used": "使用トークン数",
    "latency_ms": "応答時間（ミリ秒）"
  },
  "metadata": {
    "model_description": "モデルの説明",
    "persona_description": "ペルソナの説明",
    "persona_characteristics": ["特性1", "特性2"],
    "travel_type_description": "旅行タイプの説明",
    "target_audience": ["ターゲット1", "ターゲット2"]
  },
  "api_parameters": {
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }
}
```

### CSV形式

以下の列が含まれます：

| 列名 | 説明 |
|------|------|
| session_id | セッションID |
| experiment_id | 実験ID |
| timestamp | タイムスタンプ（ISO 8601） |
| model | モデル名 |
| model_display_name | モデル表示名 |
| persona_id | ペルソナID |
| persona_name | ペルソナ名 |
| travel_type_id | 旅行タイプID |
| travel_type_name | 旅行タイプ名 |
| prompt | プロンプト |
| response | 応答テキスト |
| tokens_used | 使用トークン数 |
| latency_ms | 応答時間（ミリ秒） |
| response_char_count | 文字数 |
| response_word_count | 単語数 |
| response_line_count | 行数 |

## 使用例

### Pythonでの読み込み

```python
import json
import pandas as pd

# JSONデータの読み込み
with open('data/examples/exp_000_gpt5_sample.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f"モデル: {data['model']['display_name']}")
    print(f"応答: {data['response'][:100]}...")

# CSVデータの読み込み
df = pd.read_csv('data/examples/sample_results.csv')
print(df[['model_display_name', 'response_char_count', 'tokens_used']])
```

## フルデータセット

完全な64パターンのデータセットを生成するには：

```bash
# 全実験を実行（約15分、APIコストが発生します）
bash scripts/run_experiment.sh
```

生成されたデータは以下に保存されます：
- **JSON**: `data/raw/`（64ファイル）
- **CSV**: `data/processed/experiment_results.csv`

## ライセンス

このサンプルデータは、プロジェクト全体と同様に[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)ライセンスで公開されています。
